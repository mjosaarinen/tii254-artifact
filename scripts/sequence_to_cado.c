/* Stream a packed GF(2) sequence into CADO's direct [S | I] coefficient file.
 *
 * Input terms are source_left_bits by source_right_bits, row-major little
 * endian.  The output selects the upper-left left_width by right_width block
 * and appends an identity block to degree zero.  Memory is O(one row), not
 * O(the complete roughly four-gigabyte coefficient series).
 */

#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t parse_u64(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0') {
        fprintf(stderr, "invalid %s\n", name);
        exit(2);
    }
    return (uint64_t)value;
}

static void write_all(FILE *stream, const unsigned char *data, size_t size) {
    if (size && fwrite(data, 1, size, stream) != size) {
        perror("write CADO direct coefficient row");
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(
            stderr,
            "usage: %s SOURCE OUTPUT TERMS SOURCE_LEFT_BITS "
            "SOURCE_RIGHT_BITS LEFT_WIDTH RIGHT_WIDTH\n",
            argv[0]);
        return 2;
    }
    const uint64_t terms = parse_u64(argv[3], "terms");
    const uint64_t source_left = parse_u64(argv[4], "source_left_bits");
    const uint64_t source_right = parse_u64(argv[5], "source_right_bits");
    const uint64_t left = parse_u64(argv[6], "left_width");
    const uint64_t right = parse_u64(argv[7], "right_width");
    if (!terms || !source_left || !source_right || !left || !right
        || left > source_left || right > source_right || (left & 63)
        || (right & 63)) {
        fputs("CADO direct stream dimensions differ\n", stderr);
        return 2;
    }
    const uint64_t source_row_bytes = ((source_right + 63) / 64) * 8;
    const uint64_t selected_bytes = right / 8;
    const uint64_t output_row_bytes = (right + left) / 8;
    if (source_row_bytes > SIZE_MAX || output_row_bytes > SIZE_MAX) {
        fputs("CADO direct stream row exceeds address space\n", stderr);
        return 2;
    }
    if (terms > UINT64_MAX / source_left
        || terms * source_left > UINT64_MAX / source_row_bytes) {
        fputs("CADO direct source size overflows\n", stderr);
        return 2;
    }
    const uint64_t expected_source_bytes = terms * source_left * source_row_bytes;
    struct stat source_stat;
    if (stat(argv[1], &source_stat) || source_stat.st_size < 0
        || (uint64_t)source_stat.st_size != expected_source_bytes) {
        fputs("CADO direct source size differs\n", stderr);
        return 1;
    }

    FILE *source = fopen(argv[1], "rb");
    if (!source) {
        perror("open CADO direct source");
        return 1;
    }
    const int output_descriptor = open(argv[2], O_WRONLY | O_CREAT | O_EXCL, 0400);
    if (output_descriptor < 0) {
        perror("create CADO direct output");
        fclose(source);
        return 1;
    }
    FILE *output = fdopen(output_descriptor, "wb");
    if (!output) {
        perror("fdopen CADO direct output");
        close(output_descriptor);
        fclose(source);
        return 1;
    }
    unsigned char *source_row = malloc((size_t)source_row_bytes);
    unsigned char *output_row = calloc((size_t)output_row_bytes, 1);
    if (!source_row || !output_row) {
        fputs("allocate CADO direct row failed\n", stderr);
        free(source_row);
        free(output_row);
        fclose(output);
        fclose(source);
        return 1;
    }
    (void)setvbuf(source, NULL, _IOFBF, 8U << 20);
    (void)setvbuf(output, NULL, _IOFBF, 8U << 20);

    for (uint64_t degree = 0; degree < terms; ++degree) {
        for (uint64_t row = 0; row < source_left; ++row) {
            if (fread(source_row, 1, (size_t)source_row_bytes, source)
                != source_row_bytes) {
                fputs("short CADO direct source read\n", stderr);
                return 1;
            }
            if (row >= left) continue;
            memset(output_row, 0, (size_t)output_row_bytes);
            memcpy(output_row, source_row, (size_t)selected_bytes);
            if (degree == 0) {
                output_row[selected_bytes + row / 8] |=
                    (unsigned char)(1U << (row & 7));
            }
            write_all(output, output_row, (size_t)output_row_bytes);
        }
    }
    if (fgetc(source) != EOF || ferror(source)) {
        fputs("trailing or failed CADO direct source read\n", stderr);
        return 1;
    }
    if (fflush(output) || fsync(output_descriptor)) {
        perror("flush CADO direct output");
        return 1;
    }
    if (fclose(output) || fclose(source)) {
        perror("close CADO direct streams");
        return 1;
    }
    free(source_row);
    free(output_row);
    printf(
        "terms=%" PRIu64 " left=%" PRIu64 " right=%" PRIu64
        " source_bytes=%" PRIu64 " output_bytes=%" PRIu64 "\n",
        terms, left, right, expected_source_bytes,
        terms * left * output_row_bytes);
    return 0;
}
