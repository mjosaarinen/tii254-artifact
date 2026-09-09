//! Independent literal `E(JQ)` replay for a reconstructed TII-254 D6 panel.
//!
//! The CUDA producer works in slice-row coordinates and checks `A Q = 0` for
//! `A = E J`.  This executable deliberately uses the independent CPU
//! Tensor--Bier operator in its original point-major coordinates.  It converts
//! the authenticated selector back to original rows, scatters `JQ`, evaluates
//! `M^T = E`, and admits the panel only when every residual bit is zero.

#[cfg(not(target_endian = "little"))]
compile_error!("the replay wire is little-endian");

use holdout_supplier_worker::operator::{BLOCK_BITS, BLOCK_WORDS, Operator, Point};
use holdout_supplier_worker::subsets::{Binom, colex_rank, subsets_of};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MANIFEST_SCHEMA: &str = "mceliecex-tii254-authenticated-d6-threshold-selector-input-v1";
const PANEL_SCHEMA_V2: &str = "mceliecex-holdout-cuda-compressed-injection-panel-v2";
const PANEL_SCHEMA_V3: &str = "mceliecex-holdout-cuda-compressed-injection-panel-v3";
const PANEL_SCHEMA_V4: &str = "mceliecex-holdout-cuda-compressed-injection-panel-v4";
const PANEL_SCHEMA_ANCHOR_UNION_V1: &str = "mceliecex-tii254-d6-anchor-panel-union-v1";
const SELECTOR_SCHEMA: &str = "mceliecex-tii254-compressed-injection-selector-v2";
const SELECTOR_LAYOUT: &str = "u32_le_cuda_slice_row_index";
const OUTPUT_SCHEMA: &str = "mceliecex-tii254-authenticated-d6-original-EJ-replay-v1";
const PASS_TERMINAL: &str = "tii254_authenticated_d6_original_EJ_replay_pass";
const REFUSAL_TERMINAL: &str = "tii254_authenticated_d6_original_EJ_replay_residual_refusal";
const PANEL_ROLE: &str = "compressed_kernel_candidates_unpromoted";
const ANCHOR_MASKED_PANEL_ROLE: &str =
    "anchor_masked_local_kernel_candidates_X=P_jQ_unpromoted";
const ANCHOR_UNION_PANEL_ROLE: &str = "anchor_masked_panel_union_unpromoted";

#[derive(Clone, Debug, PartialEq, Eq)]
struct PointRequest {
    support: u64,
    levels: Vec<usize>,
}

#[derive(Clone, Debug)]
struct OperatorRequest {
    public_sha256: String,
    k: usize,
    degree: usize,
    words: usize,
    repetitions: usize,
    seed: u64,
    emit_output: usize,
    expected_columns: u64,
    expected_rows: u64,
    expected_nonzeros: u64,
    points: Vec<PointRequest>,
}

impl OperatorRequest {
    fn same_literal_operator(&self, other: &Self) -> bool {
        self.public_sha256 == other.public_sha256
            && self.k == other.k
            && self.degree == other.degree
            && self.repetitions == other.repetitions
            && self.seed == other.seed
            && self.emit_output == other.emit_output
            && self.expected_columns == other.expected_columns
            && self.expected_rows == other.expected_rows
            && self.expected_nonzeros == other.expected_nonzeros
            && self.points == other.points
    }

    fn operator(&self) -> Operator {
        Operator::new(
            self.k,
            self.degree,
            self.points
                .iter()
                .map(|point| Point {
                    support: point.support,
                    levels: point.levels.clone(),
                })
                .collect(),
        )
    }
}

#[derive(Serialize)]
struct FileBinding {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct Shape {
    ground_set: usize,
    degree: usize,
    point_count: usize,
    form_rows: u64,
    coefficient_rows: u64,
    expected_nonzeros: u64,
    right_bits: usize,
    right_words: usize,
}

#[derive(Serialize)]
struct Timings {
    selector_and_permutation_seconds: f64,
    candidate_read_and_scatter_seconds: f64,
    source_transcript_seconds: f64,
    operator_plan_seconds: f64,
    #[serde(rename = "literal_original_E_seconds")]
    literal_original_e_seconds: f64,
    residual_transcript_seconds: f64,
    total_seconds: f64,
}

#[derive(Serialize)]
struct Checks {
    selector_manifest_and_files_bound: bool,
    base_and_width_operator_semantics_equal: bool,
    cpu_operator_shape_and_nonzeros_exact: bool,
    selector_slice_rows_form_an_injection: bool,
    selector_original_positions_digest_replayed: bool,
    candidate_header_shape_and_fnv_replayed: bool,
    #[serde(rename = "literal_original_E_applied_in_point_major_coordinates")]
    literal_original_e_applied_in_point_major_coordinates: bool,
    #[serde(rename = "every_original_EJ_residual_zero")]
    every_original_ej_residual_zero: bool,
}

#[derive(Serialize)]
struct Output {
    schema: &'static str,
    terminal: &'static str,
    claim_boundary: &'static str,
    cell_identity: String,
    operator_public_sha256: String,
    reconstruction_binding_sha256: String,
    selector_manifest: FileBinding,
    base_operator: FileBinding,
    width_operator: FileBinding,
    selector: FileBinding,
    candidate: FileBinding,
    shape: Shape,
    threads: usize,
    selector_original_positions_sha256: String,
    lifted_original_source_sha256: String,
    lifted_original_source_fnv1a64_le: String,
    lifted_original_source_nonzero_rows: u64,
    lifted_original_source_hamming_weight: u64,
    #[serde(rename = "original_EJ_residual_sha256")]
    original_ej_residual_sha256: String,
    #[serde(rename = "original_EJ_residual_fnv1a64_le")]
    original_ej_residual_fnv1a64_le: String,
    #[serde(rename = "original_EJ_residual_nonzero_rows")]
    original_ej_residual_nonzero_rows: u64,
    #[serde(rename = "original_EJ_residual_nonzero_words")]
    original_ej_residual_nonzero_words: u64,
    #[serde(rename = "original_EJ_residual_hamming_weight")]
    original_ej_residual_hamming_weight: u64,
    #[serde(rename = "every_original_EJ_residual_zero")]
    every_original_ej_residual_zero: bool,
    timings: Timings,
    checks: Checks,
}

fn lowercase_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_bytes(payload: &[u8]) -> String {
    format!("{:x}", Sha256::digest(payload))
}

fn fnv_update(mut state: u64, payload: &[u8]) -> u64 {
    for &byte in payload {
        state ^= u64::from(byte);
        state = state.wrapping_mul(1_099_511_628_211);
    }
    state
}

fn fnv_hex(payload: &[u8]) -> String {
    format!("{:016x}", fnv_update(1_469_598_103_934_665_603, payload))
}

fn read_small(path: &Path, maximum: u64) -> Result<Vec<u8>, String> {
    let size = path.metadata().map_err(|error| error.to_string())?.len();
    if size == 0 || size > maximum {
        return Err(format!("{} exceeds its input budget", path.display()));
    }
    std::fs::read(path).map_err(|error| error.to_string())
}

fn parse_operator(path: &Path) -> Result<(OperatorRequest, FileBinding), String> {
    let payload = read_small(path, 1 << 20)?;
    let binding = FileBinding {
        path: path.display().to_string(),
        size_bytes: payload.len() as u64,
        sha256: sha256_bytes(&payload),
    };
    let text = std::str::from_utf8(&payload).map_err(|error| error.to_string())?;
    let mut lines = text.lines();
    if lines.next() != Some("mceliecex-holdout-cuda-request-v1") {
        return Err("operator request schema differs".into());
    }
    let mut fields = BTreeMap::<String, String>::new();
    let mut points = Vec::new();
    for line in lines {
        let words = line.split_whitespace().collect::<Vec<_>>();
        if words.is_empty() {
            continue;
        }
        if words[0] == "point" {
            if words.len() < 4 {
                return Err("operator point is truncated".into());
            }
            let support = words[1]
                .parse::<u64>()
                .map_err(|error| format!("invalid support: {error}"))?;
            let count = words[2]
                .parse::<usize>()
                .map_err(|error| format!("invalid level count: {error}"))?;
            let levels = words[3..]
                .iter()
                .map(|value| {
                    value
                        .parse::<usize>()
                        .map_err(|error| format!("invalid level: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if levels.len() != count {
                return Err("operator point level count differs".into());
            }
            points.push(PointRequest { support, levels });
        } else if words.len() == 2 {
            if fields.insert(words[0].into(), words[1].into()).is_some() {
                return Err(format!("operator field {} repeats", words[0]));
            }
        } else {
            return Err("operator field shape differs".into());
        }
    }
    let get = |name: &str| {
        fields
            .get(name)
            .cloned()
            .ok_or_else(|| format!("operator field {name} is absent"))
    };
    let parse_usize = |name: &str| -> Result<usize, String> {
        get(name)?
            .parse()
            .map_err(|error| format!("invalid {name}: {error}"))
    };
    let parse_u64 = |name: &str| -> Result<u64, String> {
        get(name)?
            .parse()
            .map_err(|error| format!("invalid {name}: {error}"))
    };
    let expected_names = BTreeSet::from([
        "public_sha256",
        "k",
        "degree",
        "words",
        "repetitions",
        "seed",
        "emit_output",
        "expected_columns",
        "expected_rows",
        "expected_nonzeros",
        "point_count",
    ]);
    if fields.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_names {
        return Err("operator field set differs".into());
    }
    let request = OperatorRequest {
        public_sha256: get("public_sha256")?,
        k: parse_usize("k")?,
        degree: parse_usize("degree")?,
        words: parse_usize("words")?,
        repetitions: parse_usize("repetitions")?,
        seed: parse_u64("seed")?,
        emit_output: parse_usize("emit_output")?,
        expected_columns: parse_u64("expected_columns")?,
        expected_rows: parse_u64("expected_rows")?,
        expected_nonzeros: parse_u64("expected_nonzeros")?,
        points,
    };
    if !lowercase_hex(&request.public_sha256, 64)
        || request.k > 63
        || request.degree > request.k
        || request.words == 0
        || request.points.len() != parse_usize("point_count")?
    {
        return Err("operator request lies outside the replay gate".into());
    }
    Ok((request, binding))
}

fn manifest_string<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str, String> {
    let mut cursor = value;
    for name in path {
        cursor = cursor
            .get(*name)
            .ok_or_else(|| format!("manifest field {} is absent", path.join(".")))?;
    }
    cursor
        .as_str()
        .ok_or_else(|| format!("manifest field {} is not a string", path.join(".")))
}

fn manifest_u64(value: &Value, path: &[&str]) -> Result<u64, String> {
    let mut cursor = value;
    for name in path {
        cursor = cursor
            .get(*name)
            .ok_or_else(|| format!("manifest field {} is absent", path.join(".")))?;
    }
    cursor
        .as_u64()
        .ok_or_else(|| format!("manifest field {} is not an integer", path.join(".")))
}

fn validate_manifest(
    path: &Path,
    base_operator: &FileBinding,
    base: &OperatorRequest,
) -> Result<(Value, FileBinding), String> {
    let payload = read_small(path, 1 << 20)?;
    let binding = FileBinding {
        path: path.display().to_string(),
        size_bytes: payload.len() as u64,
        sha256: sha256_bytes(&payload),
    };
    let value: Value = serde_json::from_slice(&payload).map_err(|error| error.to_string())?;
    if manifest_string(&value, &["schema"])? != MANIFEST_SCHEMA
        || manifest_string(&value, &["operator", "sha256"])? != base_operator.sha256
        || manifest_string(&value, &["operator", "operator_public_sha256"])? != base.public_sha256
        || manifest_u64(&value, &["threshold", "form_rows"])? != base.expected_columns
        || manifest_u64(&value, &["threshold", "coefficient_rows"])? != base.expected_rows
        || manifest_string(&value, &["selector", "schema"])? != SELECTOR_SCHEMA
        || manifest_string(&value, &["selector", "layout"])? != SELECTOR_LAYOUT
        || manifest_u64(&value, &["selector", "count"])? != base.expected_columns
    {
        return Err("selector manifest binding differs".into());
    }
    let checks = value
        .get("checks")
        .and_then(Value::as_object)
        .ok_or("selector manifest checks are absent")?;
    if checks.is_empty() || !checks.values().all(|entry| entry == &Value::Bool(true)) {
        return Err("selector manifest does not pass every check".into());
    }
    Ok((value, binding))
}

fn slice_to_point_major(request: &OperatorRequest) -> Result<Vec<u32>, String> {
    let binom = Binom::new(request.k + 1);
    let full = (1u64 << request.k) - 1;
    let capacity = usize::try_from(request.expected_rows)
        .map_err(|_| "coefficient row count exceeds usize")?;
    let mut permutation = Vec::with_capacity(capacity);
    let mut point_base = 0u64;
    for point in &request.points {
        let support = point.support & full;
        if support != point.support {
            return Err("point support escapes the ground set".into());
        }
        let complement = full & !support;
        let mut level_offsets = BTreeMap::new();
        let mut point_rows = 0u64;
        for &level in &point.levels {
            if level > request.k || level_offsets.insert(level, point_rows).is_some() {
                return Err("point levels are invalid or repeated".into());
            }
            point_rows += binom.c(request.k, level);
        }
        for outside_size in 0..=request.degree {
            let inside_size = request.degree - outside_size;
            if outside_size > complement.count_ones() as usize
                || inside_size > support.count_ones() as usize
            {
                continue;
            }
            let outside = subsets_of(complement, outside_size);
            let active = point
                .levels
                .iter()
                .copied()
                .filter(|&level| level >= outside_size && level - outside_size <= inside_size)
                .map(|level| {
                    (
                        level,
                        subsets_of(support, level - outside_size),
                        level_offsets[&level],
                    )
                })
                .collect::<Vec<_>>();
            for &outside_mask in &outside {
                for (level, derivatives, level_offset) in &active {
                    let _ = level;
                    for &derivative in derivatives {
                        let row = point_base
                            + *level_offset
                            + colex_rank(outside_mask | derivative, &binom);
                        permutation
                            .push(u32::try_from(row).map_err(|_| "point-major row exceeds u32")?);
                    }
                }
            }
        }
        point_base += point_rows;
    }
    if point_base != request.expected_rows || permutation.len() != capacity {
        return Err("slice-to-point-major permutation length differs".into());
    }
    let mut seen = vec![0u64; capacity.div_ceil(64)];
    for &row in &permutation {
        let index = row as usize;
        if index >= capacity || (seen[index / 64] >> (index % 64)) & 1 != 0 {
            return Err("slice-to-point-major map is not a permutation".into());
        }
        seen[index / 64] |= 1u64 << (index % 64);
    }
    if (0..capacity).any(|index| (seen[index / 64] >> (index % 64)) & 1 == 0) {
        return Err("slice-to-point-major map is incomplete".into());
    }
    Ok(permutation)
}

fn read_selector(
    path: &Path,
    manifest: &Value,
    request: &OperatorRequest,
    permutation: &[u32],
) -> Result<(Vec<u32>, FileBinding, String), String> {
    let expected_bytes = request
        .expected_columns
        .checked_mul(4)
        .ok_or("selector byte count overflows")?;
    let payload = read_small(path, expected_bytes)?;
    if payload.len() as u64 != expected_bytes {
        return Err("selector payload size differs".into());
    }
    let binding = FileBinding {
        path: path.display().to_string(),
        size_bytes: payload.len() as u64,
        sha256: sha256_bytes(&payload),
    };
    if binding.sha256 != manifest_string(manifest, &["selector", "payload_sha256"])?
        || binding.sha256 != manifest_string(manifest, &["selector", "payload", "sha256"])?
        || binding.size_bytes != manifest_u64(manifest, &["selector", "payload", "size_bytes"])?
        || fnv_hex(&payload) != manifest_string(manifest, &["selector", "payload_fnv1a64_le"])?
    {
        return Err("selector payload binding differs".into());
    }
    let mut original = Vec::with_capacity(payload.len() / 4);
    let mut original_digest = Sha256::new();
    let mut seen = vec![0u64; permutation.len().div_ceil(64)];
    for bytes in payload.chunks_exact(4) {
        let slice = u32::from_le_bytes(bytes.try_into().unwrap()) as usize;
        if slice >= permutation.len() || (seen[slice / 64] >> (slice % 64)) & 1 != 0 {
            return Err("selector slice rows do not form an injection".into());
        }
        seen[slice / 64] |= 1u64 << (slice % 64);
        let row = permutation[slice];
        original_digest.update(row.to_le_bytes());
        original.push(row);
    }
    let digest = format!("{:x}", original_digest.finalize());
    if digest != manifest_string(manifest, &["selector", "original_positions_sha256"])? {
        return Err("selector original-position digest differs".into());
    }
    Ok((original, binding, digest))
}

fn read_line_hashed(reader: &mut impl BufRead, digest: &mut Sha256) -> Result<Vec<u8>, String> {
    let mut value = Vec::new();
    let count = reader
        .read_until(b'\n', &mut value)
        .map_err(|error| error.to_string())?;
    if count == 0 || value.last() != Some(&b'\n') {
        return Err("candidate header is truncated".into());
    }
    digest.update(&value);
    Ok(value)
}

struct CandidateRead {
    binding: FileBinding,
    reconstruction_binding: String,
    source: Vec<u64>,
    source_nonzero_rows: u64,
    source_hamming_weight: u64,
}

fn read_candidate_and_scatter(
    path: &Path,
    maximum_bytes: u64,
    request: &OperatorRequest,
    original_rows: &[u32],
) -> Result<CandidateRead, String> {
    let size = path.metadata().map_err(|error| error.to_string())?.len();
    if size == 0 || size > maximum_bytes {
        return Err("candidate panel is empty or exceeds its byte ceiling".into());
    }
    let mut reader = BufReader::with_capacity(
        8 << 20,
        File::open(path).map_err(|error| error.to_string())?,
    );
    let mut digest = Sha256::new();
    let panel_schema = read_line_hashed(&mut reader, &mut digest)?;
    if panel_schema != format!("{PANEL_SCHEMA_V2}\n").as_bytes()
        && panel_schema != format!("{PANEL_SCHEMA_V3}\n").as_bytes()
        && panel_schema != format!("{PANEL_SCHEMA_V4}\n").as_bytes()
        && panel_schema != format!("{PANEL_SCHEMA_ANCHOR_UNION_V1}\n").as_bytes()
    {
        return Err("candidate panel schema differs".into());
    }
    let mut fields = BTreeMap::<String, String>::new();
    loop {
        let raw = read_line_hashed(&mut reader, &mut digest)?;
        if raw == b"data_le\n" {
            break;
        }
        let line = std::str::from_utf8(&raw)
            .map_err(|error| error.to_string())?
            .trim_end_matches('\n');
        let (name, value) = line
            .split_once(' ')
            .ok_or("candidate header field is malformed")?;
        if fields.insert(name.into(), value.into()).is_some() {
            return Err(format!("candidate header repeats {name}"));
        }
    }
    let expected_fields = BTreeSet::from([
        "reconstruction_binding_sha256",
        "role",
        "rows",
        "right_bits",
        "panel_words",
        "panel_fnv1a64_le",
    ]);
    if fields.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_fields {
        return Err("candidate header field set differs".into());
    }
    let field = |name: &str| {
        fields
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("candidate field {name} is absent"))
    };
    let integer = |name: &str| -> Result<u64, String> {
        field(name)?
            .parse()
            .map_err(|error| format!("invalid candidate {name}: {error}"))
    };
    let rows = integer("rows")?;
    let right_bits = integer("right_bits")? as usize;
    let words = right_bits / 64;
    let panel_words = integer("panel_words")?;
    let reconstruction_binding = field("reconstruction_binding_sha256")?.to_string();
    let expected_role = if panel_schema == format!("{PANEL_SCHEMA_ANCHOR_UNION_V1}\n").as_bytes() {
        ANCHOR_UNION_PANEL_ROLE
    } else if panel_schema == format!("{PANEL_SCHEMA_V4}\n").as_bytes() {
        ANCHOR_MASKED_PANEL_ROLE
    } else {
        PANEL_ROLE
    };
    if field("role")? != expected_role
        || !lowercase_hex(&reconstruction_binding, 64)
        || rows != request.expected_columns
        || rows != original_rows.len() as u64
        || right_bits != BLOCK_BITS
        || right_bits % 64 != 0
        || words != BLOCK_WORDS
        || panel_words
            != rows
                .checked_mul(words as u64)
                .ok_or("panel size overflows")?
    {
        return Err("candidate panel shape or role differs".into());
    }
    let source_words = usize::try_from(request.expected_rows)
        .map_err(|_| "coefficient rows exceed usize")?
        .checked_mul(BLOCK_WORDS)
        .ok_or("source panel size overflows")?;
    let mut source = vec![0u64; source_words];
    let mut payload_fnv = 1_469_598_103_934_665_603u64;
    let mut raw = [0u8; BLOCK_WORDS * 8];
    let mut nonzero_rows = 0u64;
    let mut hamming = 0u64;
    for &original in original_rows {
        reader
            .read_exact(&mut raw)
            .map_err(|error| error.to_string())?;
        digest.update(raw);
        payload_fnv = fnv_update(payload_fnv, &raw);
        let start = original as usize * BLOCK_WORDS;
        let mut nonzero = false;
        for (destination, bytes) in source[start..start + BLOCK_WORDS]
            .iter_mut()
            .zip(raw.chunks_exact(8))
        {
            *destination = u64::from_le_bytes(bytes.try_into().unwrap());
            nonzero |= *destination != 0;
            hamming += u64::from(destination.count_ones());
        }
        nonzero_rows += u64::from(nonzero);
    }
    let mut trailing = [0u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| error.to_string())?
        != 0
    {
        return Err("candidate panel has trailing bytes".into());
    }
    if format!("{payload_fnv:016x}") != field("panel_fnv1a64_le")? {
        return Err("candidate panel FNV differs".into());
    }
    Ok(CandidateRead {
        binding: FileBinding {
            path: path.display().to_string(),
            size_bytes: size,
            sha256: format!("{:x}", digest.finalize()),
        },
        reconstruction_binding,
        source,
        source_nonzero_rows: nonzero_rows,
        source_hamming_weight: hamming,
    })
}

fn words_as_bytes(words: &[u64]) -> &[u8] {
    // Every word is initialized, the executable is little-endian, and the
    // borrowed byte view does not outlive the words.  This avoids a second
    // multi-GiB transcript allocation.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words)) }
}

fn write_output(path: &Path, value: &Output) -> Result<(), String> {
    if path.exists() {
        return Err("output result already exists".into());
    }
    let payload = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| error.to_string())?,
    );
    output
        .write_all(&payload)
        .map_err(|error| error.to_string())?;
    output.write_all(b"\n").map_err(|error| error.to_string())?;
    output.flush().map_err(|error| error.to_string())?;
    output
        .get_ref()
        .sync_all()
        .map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::to_string(value).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn replay(
    manifest_path: &Path,
    base_operator_path: &Path,
    width_operator_path: &Path,
    selector_path: &Path,
    candidate_path: &Path,
    maximum_bytes: u64,
    threads: usize,
    output_path: &Path,
) -> Result<bool, String> {
    let total_started = Instant::now();
    if threads == 0 || threads > 512 {
        return Err("thread count lies outside 1..=512".into());
    }
    let (base, base_binding) = parse_operator(base_operator_path)?;
    let (width, width_binding) = parse_operator(width_operator_path)?;
    if !base.same_literal_operator(&width) || base.words != 1 || width.words * 64 != BLOCK_BITS {
        return Err("base and width-specific literal operators differ".into());
    }
    let (manifest, manifest_binding) = validate_manifest(manifest_path, &base_binding, &base)?;
    let selector_started = Instant::now();
    let permutation = slice_to_point_major(&width)?;
    let (original_rows, selector_binding, original_digest) =
        read_selector(selector_path, &manifest, &width, &permutation)?;
    drop(permutation);
    let selector_seconds = selector_started.elapsed().as_secs_f64();

    let mut operator = width.operator();
    if operator.n_cols != width.expected_columns
        || operator.n_rows != width.expected_rows
        || operator.nnz() != width.expected_nonzeros
    {
        return Err("CPU literal operator shape or nonzero count differs".into());
    }

    let candidate_started = Instant::now();
    let candidate =
        read_candidate_and_scatter(candidate_path, maximum_bytes, &width, &original_rows)?;
    let candidate_seconds = candidate_started.elapsed().as_secs_f64();
    drop(original_rows);

    let source_transcript_started = Instant::now();
    let source_bytes = words_as_bytes(&candidate.source);
    let source_sha = sha256_bytes(source_bytes);
    let source_fnv = fnv_hex(source_bytes);
    let source_transcript_seconds = source_transcript_started.elapsed().as_secs_f64();

    let plan_started = Instant::now();
    operator.prepare(threads);
    let plan_seconds = plan_started.elapsed().as_secs_f64();
    let mut residual = vec![0u64; width.expected_columns as usize * BLOCK_WORDS];
    let apply_started = Instant::now();
    operator.apply_transpose(&candidate.source, &mut residual, threads);
    let apply_seconds = apply_started.elapsed().as_secs_f64();
    drop(candidate.source);

    let residual_started = Instant::now();
    let mut residual_nonzero_rows = 0u64;
    let mut residual_nonzero_words = 0u64;
    let mut residual_hamming = 0u64;
    for row in residual.chunks_exact(BLOCK_WORDS) {
        let nonzero = row.iter().any(|&word| word != 0);
        residual_nonzero_rows += u64::from(nonzero);
        for &word in row {
            residual_nonzero_words += u64::from(word != 0);
            residual_hamming += u64::from(word.count_ones());
        }
    }
    let residual_bytes = words_as_bytes(&residual);
    let residual_sha = sha256_bytes(residual_bytes);
    let residual_fnv = fnv_hex(residual_bytes);
    let residual_seconds = residual_started.elapsed().as_secs_f64();
    let zero = residual_nonzero_words == 0;

    let output = Output {
        schema: OUTPUT_SCHEMA,
        terminal: if zero {
            PASS_TERMINAL
        } else {
            REFUSAL_TERMINAL
        },
        claim_boundary: "independent literal original-coordinate E(JX) replay of one finite reconstructed panel X; a nonzero residual refuses admission, while a zero residual proves only that the supplied vectors are public relations and does not establish canonical containment, completeness, locator recovery, or a key",
        cell_identity: manifest_string(&manifest, &["cell_identity"])?.to_string(),
        operator_public_sha256: width.public_sha256.clone(),
        reconstruction_binding_sha256: candidate.reconstruction_binding,
        selector_manifest: manifest_binding,
        base_operator: base_binding,
        width_operator: width_binding,
        selector: selector_binding,
        candidate: candidate.binding,
        shape: Shape {
            ground_set: width.k,
            degree: width.degree,
            point_count: width.points.len(),
            form_rows: width.expected_columns,
            coefficient_rows: width.expected_rows,
            expected_nonzeros: width.expected_nonzeros,
            right_bits: BLOCK_BITS,
            right_words: BLOCK_WORDS,
        },
        threads,
        selector_original_positions_sha256: original_digest,
        lifted_original_source_sha256: source_sha,
        lifted_original_source_fnv1a64_le: source_fnv,
        lifted_original_source_nonzero_rows: candidate.source_nonzero_rows,
        lifted_original_source_hamming_weight: candidate.source_hamming_weight,
        original_ej_residual_sha256: residual_sha,
        original_ej_residual_fnv1a64_le: residual_fnv,
        original_ej_residual_nonzero_rows: residual_nonzero_rows,
        original_ej_residual_nonzero_words: residual_nonzero_words,
        original_ej_residual_hamming_weight: residual_hamming,
        every_original_ej_residual_zero: zero,
        timings: Timings {
            selector_and_permutation_seconds: selector_seconds,
            candidate_read_and_scatter_seconds: candidate_seconds,
            source_transcript_seconds,
            operator_plan_seconds: plan_seconds,
            literal_original_e_seconds: apply_seconds,
            residual_transcript_seconds: residual_seconds,
            total_seconds: total_started.elapsed().as_secs_f64(),
        },
        checks: Checks {
            selector_manifest_and_files_bound: true,
            base_and_width_operator_semantics_equal: true,
            cpu_operator_shape_and_nonzeros_exact: true,
            selector_slice_rows_form_an_injection: true,
            selector_original_positions_digest_replayed: true,
            candidate_header_shape_and_fnv_replayed: true,
            literal_original_e_applied_in_point_major_coordinates: true,
            every_original_ej_residual_zero: zero,
        },
    };
    write_output(output_path, &output)?;
    Ok(zero)
}

fn main_result() -> Result<bool, String> {
    let arguments = env::args().collect::<Vec<_>>();
    if arguments.len() != 10 || arguments[1] != "replay" {
        return Err("usage: tii254-d6-original-ej-replay replay SELECTOR_MANIFEST.json BASE_OPERATOR.txt WIDTH_OPERATOR.txt SELECTOR.u32le CANDIDATES.bin MAX_BYTES THREADS NEW_RESULT.json".into());
    }
    let maximum_bytes = arguments[7]
        .parse::<u64>()
        .map_err(|error| format!("invalid maximum bytes: {error}"))?;
    let threads = arguments[8]
        .parse::<usize>()
        .map_err(|error| format!("invalid threads: {error}"))?;
    replay(
        &PathBuf::from(&arguments[2]),
        &PathBuf::from(&arguments[3]),
        &PathBuf::from(&arguments[4]),
        &PathBuf::from(&arguments[5]),
        &PathBuf::from(&arguments[6]),
        maximum_bytes,
        threads,
        &PathBuf::from(&arguments[9]),
    )
}

fn main() {
    match main_result() {
        Ok(true) => {}
        Ok(false) => std::process::exit(2),
        Err(error) => {
            eprintln!("tii254-d6-original-ej-replay: {error}");
            std::process::exit(1);
        }
    }
}
