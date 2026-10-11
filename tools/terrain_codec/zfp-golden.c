/* Regenerates the recorded float-1 reversible goldens with upstream 1.0.1.
 * Run from an unmodified checkout at f40868a6a1c190c802e7d8b5987064f044bf7812:
 * clang -std=c99 -O2 -fwrapv -Iinclude -Isrc PATH/zfp-golden.c
 *       src/encode1f.c src/decode1f.c src/bitstream.c -o zfp-golden
 * Input: four hexadecimal u32 words per line. Output: one JSON object per
 * line, with the complete block bits, bytes and upstream-decoded words.
 * C is only a provenance tool; production and acceptance runners use Rust.
 */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include "zfp.h"

int main(void) {
  uint32_t words[4];
  while (scanf("%" SCNx32 " %" SCNx32 " %" SCNx32 " %" SCNx32,
               words, words + 1, words + 2, words + 3) == 4) {
    unsigned char buffer[64] = {0};
    float input[4], decoded[4];
    uint32_t output[4];
    memcpy(input, words, sizeof input);
    bitstream* stream = stream_open(buffer, sizeof buffer);
    zfp_stream zfp = {ZFP_MIN_BITS, ZFP_MAX_BITS, ZFP_MAX_PREC,
                      ZFP_MIN_EXP - 1, stream, {zfp_exec_serial, NULL}};
    size_t bits = zfp_encode_block_float_1(&zfp, input);
    stream_flush(stream);
    stream_rewind(stream);
    size_t consumed = zfp_decode_block_float_1(&zfp, decoded);
    memcpy(output, decoded, sizeof output);
    if (consumed != bits || memcmp(words, output, sizeof words)) return 3;
    printf("{\"input\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32
           "],\"bits\":%zu,\"bytes\":[", words[0], words[1], words[2], words[3], bits);
    for (size_t i = 0; i < (bits + 7) / 8; i++)
      printf("%s%u", i ? "," : "", buffer[i]);
    printf("],\"decoded\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32 "]}\n",
           output[0], output[1], output[2], output[3]);
    stream_close(stream);
  }
  return 0;
}
