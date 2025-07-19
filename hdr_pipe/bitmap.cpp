#include "bitmap.h"
#include "mtwister.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

enum Compression {
    RGB = 0,
    RLE8,
    RLE4,
    BITFIELDS,
    JPEG,
    PNG,
};

#pragma pack(push, 1)

//2 bytes
struct FileMagic {
    uint8_t num0, num1;
};

//12 bytes
struct FileHeader {
    uint32_t fileSize;
    uint16_t creators[2];
    uint32_t dataOffset;
};

//40 bytes, all windows versions since 3.0
struct DibHeader {
    uint32_t headerSize;
    int32_t width, height;
    uint16_t numPlanes, bitsPerPixel;
    uint32_t compression;
    uint32_t dataSize;
    int32_t hPixelsPer, vPixelsPer;  //horizontal and vertical pixels-per-meter
    uint32_t numPalColors, numImportantColors;
};


bool exportBMP(const std::vector<uint8_t>& imageData, int w, int h, const char* filename) {
    if (imageData.size() != (w * h * 3)) {
        printf("%s - invalid data size\n", __FUNCTION__);
        return false;
    }

    // Add ".bmp" if not already there
    char file[256];
    strncpy(file, filename, sizeof(file));
    file[sizeof(file) - 1] = '\0';
    if (strstr(file, ".bmp") == nullptr)
        strcat(file, ".bmp");

    FILE* fp = fopen(file, "wb");
    if (!fp) {
        perror("fopen");
        return false;
    }

    // Padding per row (rows must be aligned to 4 bytes)
    int rowSize = w * 3;
    int padding = (4 - (rowSize % 4)) % 4;
    int paddedRowSize = rowSize + padding;
    int dataSize = paddedRowSize * h;

    // --- File Header ---
    FileMagic magic = { 'B', 'M' };
    fwrite(&magic, 2, 1, fp);

    FileHeader fileHeader;
    fileHeader.fileSize = 14 + 40 + dataSize; // file header + DIB header + pixel data
    fileHeader.creators[0] = fileHeader.creators[1] = 0;
    fileHeader.dataOffset = 14 + 40;
    fwrite(&fileHeader, sizeof(fileHeader), 1, fp);

    // --- DIB Header ---
    DibHeader dibHeader;
    dibHeader.headerSize = 40;
    dibHeader.width = w;
    dibHeader.height = h;
    dibHeader.numPlanes = 1;
    dibHeader.bitsPerPixel = 24;
    dibHeader.compression = RGB;
    dibHeader.dataSize = dataSize;
    dibHeader.hPixelsPer = dibHeader.vPixelsPer = 1000;
    dibHeader.numPalColors = dibHeader.numImportantColors = 0;
    fwrite(&dibHeader, sizeof(DibHeader), 1, fp);

    // --- Pixel Data (bottom-up) ---
    for (int y = h - 1; y >= 0; y--) {
        const uint8_t* row = &imageData[y * w * 3];
        fwrite(row, 1, rowSize, fp);
        for (int i = 0; i < padding; i++)
            fputc(0x00, fp); // zero padding
    }

    fclose(fp);
    printf("%s - bitmap file '%s' created\n", __FUNCTION__, file);
    return true;
}

bool importBMP(std::vector<uint8_t>& imageData, int& w, int& h, const char* filename) {
    imageData.clear();
    w = h = 0;

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("%s - couldn't open BMP file '%s'\n", __FUNCTION__, filename);
        return false;
    }

    // --- Read and validate FileMagic ---
    FileMagic magic;
    fread(&magic, 2, 1, fp);
    if (magic.num0 != 'B' || magic.num1 != 'M') {
        printf("%s - not a BMP file\n", __FUNCTION__);
        fclose(fp);
        return false;
    }

    // --- Read FileHeader and DIB Header ---
    FileHeader fileHeader;
    fread(&fileHeader, sizeof(FileHeader), 1, fp);

    DibHeader dibHeader;
    fread(&dibHeader, sizeof(DibHeader), 1, fp);

    if (dibHeader.bitsPerPixel != 24 || dibHeader.compression != RGB) {
        printf("%s - only 24-bit uncompressed BMP supported\n", __FUNCTION__);
        fclose(fp);
        return false;
    }

    w = dibHeader.width;
    h = dibHeader.height;

    // Padding per row
    int rowSize = w * 3;
    int padding = (4 - (rowSize % 4)) % 4;
    int paddedRowSize = rowSize + padding;

    imageData.resize(w * h * 3);

    // Seek to pixel data
    fseek(fp, fileHeader.dataOffset, SEEK_SET);

    // Read pixel data (bottom-up)
    for (int y = h - 1; y >= 0; y--) {
        uint8_t* row = &imageData[y * w * 3];
        fread(row, 1, rowSize, fp);
        fseek(fp, padding, SEEK_CUR); // skip padding
    }

    fclose(fp);
    printf("%s - BMP file '%s' loaded\n", __FUNCTION__, filename);
    return true;
}

static uint32_t hashit(const char* key) {
    uint32_t h = 2166136261u;
    while (*key)
        h = (h ^ (uint8_t)*key++) * 16777619;
    return h;
}

void encryptBMP(const std::vector<uint8_t>& imageData, const char* key, std::vector<uint8_t>& imageEncrypted) {
    uint32_t seed = hashit(key);
    MTRand mt_rand = seedRand(seed);

    uint32_t numBytes = imageData.size();
    imageEncrypted.resize(numBytes);

    std::vector<uint32_t> perm(numBytes);
    for (uint32_t i = 0; i < numBytes; ++i)
        perm[i] = i;

    // Step 2: Fisher–Yates shuffle with MT
    for (size_t i = numBytes - 1; i > 0; --i) {
        size_t j = genRandLong(&mt_rand) % (i + 1);
        std::swap(perm[i], perm[j]);
    }

    // Step 3: Apply permutation to input data
    for (size_t i = 0; i < numBytes; ++i) {
        imageEncrypted[perm[i]] = imageData[i];
    }
}

void genPermTables(const char* key, size_t numValues, std::vector<uint32_t>& perm, std::vector<uint32_t>& inversePerm) {
    uint32_t seed = hashit(key);
    MTRand mt_rand = seedRand(seed);

    perm.resize(numValues);
    inversePerm.resize(numValues);

    // Generate forward permutation using Fisher-Yates
    for (uint32_t i = 0; i < numValues; ++i)
        perm[i] = i;

    for (size_t i = numValues - 1; i > 0; --i) {
        size_t j = genRandLong(&mt_rand) % (i + 1);
        std::swap(perm[i], perm[j]);
    }

    // Invert the permutation
    for (size_t i = 0; i < numValues; ++i) {
        inversePerm[perm[i]] = i;
    }
}

void decryptBMP(const std::vector<uint8_t>& imageEncrypted, std::vector<uint8_t>& imageData, const std::vector<uint32_t>& perm) {
    imageData.resize(imageEncrypted.size());
    for (size_t i = 0; i < imageEncrypted.size(); ++i) {
        //imageData[inversePerm[i]] = imageEncrypted[i];
        imageData[i] = imageEncrypted[perm[i]];
    }
}
