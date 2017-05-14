#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "Config.h"
#include "image.h"
#include "partition.h"
#include <memory>
#include <vector>
#include "sse_debug.h"

namespace Frac {
	class ImageClassifier {
	public:
		virtual ~ImageClassifier() {}
		virtual bool compare(const PartitionItemPtr& targetItem, const PartitionItemPtr& sourceItem) const {
			return this->compare(targetItem->image(), sourceItem->image());
		}
		virtual bool compare(const Image& a, const Image& b) const = 0;
	};

	class CombinedClassifier : public ImageClassifier {
	public:
		CombinedClassifier() {

		}
		explicit CombinedClassifier(ImageClassifier* c1) {
			_classifiers.emplace_back(c1);
		}
		CombinedClassifier(ImageClassifier* c1, ImageClassifier* c2) {
			_classifiers.emplace_back(c1);
			_classifiers.emplace_back(c2);
		}
		CombinedClassifier(ImageClassifier* c1, ImageClassifier* c2, ImageClassifier* c3) {
			_classifiers.emplace_back(c1);
			_classifiers.emplace_back(c2);
			_classifiers.emplace_back(c3);
		}
		bool compare(const PartitionItemPtr& a, const PartitionItemPtr& b) const override {
			for (size_t i = 0; i < _classifiers.size(); ++i) {
				auto p = _classifiers.data() + i;
				if (!(*p)->compare(a, b))
					return false;
			}
			return true;
		}
		bool compare(const Image& a, const Image& b) const override {
			for (size_t i = 0; i < _classifiers.size(); ++i) {
				auto p = _classifiers.data() + i;
				if (!(*p)->compare(a, b))
					return false;
			}
			return true;
		}
	private:
		std::vector<std::unique_ptr<ImageClassifier>> _classifiers;
	};

	class DummyClassifier : public ImageClassifier {
	public:
		bool compare(const Image&, const Image&) const override {
			return true;
		}
	};

	class TextureClassifier : public ImageClassifier {
	public:
		bool compare(const Image& a, const Image& b) const override {
			const auto va = ImageStatistics::variance(a);
			const auto vb = ImageStatistics::variance(b);
			return this->isFlat(va) == this->isFlat(vb);
		}
		bool isFlat(const Image& image) const noexcept {
			return isFlat(ImageStatistics::variance(image));
		}
	private:
		bool isFlat(const double var) const noexcept {
			return var < 2500.0;
		}
	};

	class ThresholdClassifier : public ImageClassifier {
	public:
		bool compare(const Image& a, const Image& b) const override {
			const auto va = ImageStatistics::variance(a);
			const auto vb = ImageStatistics::variance(b);
			return this->category(va) == this->category(vb);
		}
	protected:
		int category(const double var) const noexcept {
			return var < 2000 ? 0 : var < 4000 ? 1 : var < 6000 ? 2 : 3;
		}
	};

	class BrightnessBlockClassifier : public ImageClassifier {
	public:
		bool compare(const Image& a, const Image& b) const override {
			int typeA = a.cache() ? (int)a.cache()->get(ImageData::KeyBlockTypeBrightness, -1) : -1;
			int typeB = b.cache() ? (int)b.cache()->get(ImageData::KeyBlockTypeBrightness, -1) : -1;
			if (typeA == -1) {
				typeA = BrightnessBlockClassifier::getCategory(a);
				if (a.cache())
					a.cache()->put(ImageData::KeyBlockTypeBrightness, typeA);
			}
			if (typeB == -1) {
				typeB = BrightnessBlockClassifier::getCategory(b);
				if (b.cache())
					b.cache()->put(ImageData::KeyBlockTypeBrightness, typeB);
			}
			return typeA == typeB;
		}
	private:
		static int getCategory(const Image& image) {
			if (image.width() == 2) {
				const auto data = image.data()->get();
				return category(data[0], data[1], data[0 + image.stride()], data[1 + image.stride()]);
			}
#ifdef FRAC_WITH_AVX
			if (image.width() == 4) {
				const auto row0 = image.data()->get();
				const auto row1 = row0 + image.stride();
				const auto row2 = row0 + 2 * image.stride();
				const auto row3 = row0 + 3 * image.stride();

				__m128i row01 = _mm_set_epi32(0, 0, *(int32_t*)(row1), *(int32_t*)row0); // [0] [0] [row1] [row0]
				row01 = _mm_unpacklo_epi8(row01, _mm_setzero_si128()); // [row1_3, row1_2] [row1_1, row1_0] [row0_3, row0_2] [row0_1, row0_0]
				
				__m128i shuffle = _mm_shuffle_epi32(row01, _MM_SHUFFLE(1, 0, 3, 2));
				shuffle = _mm_add_epi16(shuffle, row01);
				
				row01 = _mm_shufflehi_epi16(shuffle, _MM_SHUFFLE(2, 3, 0, 1));
				row01 = _mm_add_epi16(row01, shuffle);
				
				const double a1 = _mm_extract_epi16(row01, 5) / 4.0;
				const double a2 = _mm_extract_epi16(row01, 7) / 4.0;

				row01 = _mm_set_epi32(0, 0, *(int32_t*)(row3), *(int32_t*)row2);
				row01 = _mm_unpacklo_epi8(row01, _mm_setzero_si128());

				shuffle = _mm_shuffle_epi32(row01, _MM_SHUFFLE(1, 0, 3, 2));
				shuffle = _mm_add_epi16(shuffle, row01);

				row01 = _mm_shufflehi_epi16(shuffle, _MM_SHUFFLE(2, 3, 0, 1));
				row01 = _mm_add_epi16(row01, shuffle);

				const double a3 = _mm_extract_epi16(row01, 5) / 4.0;
				const double a4 = _mm_extract_epi16(row01, 7) / 4.0;

				return category(a1, a2, a3, a4);
			} else if (image.width() == 8) {
				const auto row0 = image.data()->get();
				const auto row1 = row0 + image.stride();
				const auto row2 = row0 + 2 * image.stride();
				const auto row3 = row0 + 3 * image.stride();
				const auto row4 = row0 + 4 * image.stride();
				const auto row5 = row0 + 5 * image.stride();
				const auto row6 = row0 + 6 * image.stride();
				const auto row7 = row0 + 7 * image.stride();

				__m128i row_sse = _mm_unpacklo_epi8(_mm_set_epi64x(0, *(uint64_t*)row0), _mm_setzero_si128());
				
				row_sse = _mm_add_epi16(row_sse, _mm_unpacklo_epi8(_mm_set_epi64x(0, *(uint64_t*)row1), _mm_setzero_si128()));
				row_sse = _mm_add_epi16(row_sse, _mm_unpacklo_epi8(_mm_set_epi64x(0, *(uint64_t*)row2), _mm_setzero_si128()));
				row_sse = _mm_add_epi16(row_sse, _mm_unpacklo_epi8(_mm_set_epi64x(0, *(uint64_t*)row3), _mm_setzero_si128()));

				__m128i shuffle = _mm_shuffle_epi32(row_sse, _MM_SHUFFLE(2, 3, 0, 1));
				shuffle = _mm_add_epi16(shuffle, row_sse);

				row_sse = _mm_shufflehi_epi16(shuffle, _MM_SHUFFLE(2, 3, 0, 1));
				row_sse = _mm_shufflelo_epi16(row_sse, _MM_SHUFFLE(2, 3, 0, 1));
				row_sse = _mm_add_epi16(row_sse, shuffle);

				const double a1 = _mm_extract_epi16(row_sse, 0) / 16.0;
				const double a2 = _mm_extract_epi16(row_sse, 7) / 16.0;

				row_sse = _mm_unpacklo_epi8(_mm_set_epi64x(0, *(uint64_t*)row4), _mm_setzero_si128());
				
				row_sse = _mm_add_epi16(row_sse, _mm_unpacklo_epi8(_mm_set_epi64x(0, *(uint64_t*)row5), _mm_setzero_si128()));
				row_sse = _mm_add_epi16(row_sse, _mm_unpacklo_epi8(_mm_set_epi64x(0, *(uint64_t*)row6), _mm_setzero_si128()));
				row_sse = _mm_add_epi16(row_sse, _mm_unpacklo_epi8(_mm_set_epi64x(0, *(uint64_t*)row7), _mm_setzero_si128()));

				shuffle = _mm_shuffle_epi32(row_sse, _MM_SHUFFLE(2, 3, 0, 1));
				shuffle = _mm_add_epi16(shuffle, row_sse);

				row_sse = _mm_shufflehi_epi16(shuffle, _MM_SHUFFLE(2, 3, 0, 1));
				row_sse = _mm_shufflelo_epi16(row_sse, _MM_SHUFFLE(2, 3, 0, 1));
				row_sse = _mm_add_epi16(row_sse, shuffle);

				const double a3 = _mm_extract_epi16(row_sse, 0) / 16.0;
				const double a4 = _mm_extract_epi16(row_sse, 7) / 16.0;
				return category(a1, a2, a3, a4);
			} else if (image.width() == 16) {
				const auto row0 = image.data()->get();
				const auto row1 = row0 + image.stride();
				const auto row2 = row0 + 2 * image.stride();
				const auto row3 = row0 + 3 * image.stride();
				const auto row4 = row0 + 4 * image.stride();
				const auto row5 = row0 + 5 * image.stride();
				const auto row6 = row0 + 6 * image.stride();
				const auto row7 = row0 + 7 * image.stride();
				const auto row8 = row0 + 8 * image.stride();
				const auto row9 = row0 + 9 * image.stride();
				const auto row10 = row0 + 10 * image.stride();
				const auto row11 = row0 + 11 * image.stride();
				const auto row12 = row0 + 12 * image.stride();
				const auto row13 = row0 + 13 * image.stride();
				const auto row14 = row0 + 14 * image.stride();
				const auto row15 = row0 + 15 * image.stride();

				__m256i row_avx = _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row0 + 8), 0, *(uint64_t*)row0), _mm256_setzero_si256());

				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row1 + 8), 0, *(uint64_t*)row1), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row2 + 8), 0, *(uint64_t*)row2), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row3 + 8), 0, *(uint64_t*)row3), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row4 + 8), 0, *(uint64_t*)row4), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row5 + 8), 0, *(uint64_t*)row5), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row6 + 8), 0, *(uint64_t*)row6), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row7 + 8), 0, *(uint64_t*)row7), _mm256_setzero_si256()));

				__m256i shuffle = _mm256_shuffle_epi32(row_avx, _MM_SHUFFLE(2, 3, 0, 1));
				shuffle = _mm256_add_epi16(shuffle, row_avx);
				row_avx = _mm256_shuffle_epi32(shuffle, _MM_SHUFFLE(0, 2, 1, 3));
				shuffle = _mm256_add_epi16(row_avx, shuffle);

				row_avx = _mm256_shufflehi_epi16(shuffle, _MM_SHUFFLE(2, 3, 0, 1));
				row_avx = _mm256_shufflelo_epi16(row_avx, _MM_SHUFFLE(2, 3, 0, 1));
				row_avx = _mm256_add_epi16(row_avx, shuffle);

				const double a1 = _mm_extract_epi16(_mm256_extracti128_si256(row_avx, 0), 0) / 64.0;
				const double a2 = _mm_extract_epi16(_mm256_extracti128_si256(row_avx, 1), 0) / 64.0;


				row_avx = _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row8 + 8), 0, *(uint64_t*)row8), _mm256_setzero_si256());
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row9 + 8), 0, *(uint64_t*)row9), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row10 + 8), 0, *(uint64_t*)row10), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row11 + 8), 0, *(uint64_t*)row11), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row12 + 8), 0, *(uint64_t*)row12), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row13 + 8), 0, *(uint64_t*)row13), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row14 + 8), 0, *(uint64_t*)row14), _mm256_setzero_si256()));
				row_avx = _mm256_add_epi16(row_avx, _mm256_unpacklo_epi8(_mm256_set_epi64x(0, *(uint64_t*)(row15 + 8), 0, *(uint64_t*)row15), _mm256_setzero_si256()));

				shuffle = _mm256_shuffle_epi32(row_avx, _MM_SHUFFLE(2, 3, 0, 1));
				shuffle = _mm256_add_epi16(shuffle, row_avx);
				row_avx = _mm256_shuffle_epi32(shuffle, _MM_SHUFFLE(0, 2, 1, 3));
				shuffle = _mm256_add_epi16(row_avx, shuffle);

				row_avx = _mm256_shufflehi_epi16(shuffle, _MM_SHUFFLE(2, 3, 0, 1));
				row_avx = _mm256_shufflelo_epi16(row_avx, _MM_SHUFFLE(2, 3, 0, 1));
				row_avx = _mm256_add_epi16(row_avx, shuffle);

				const double a3 = _mm_extract_epi16(_mm256_extracti128_si256(row_avx, 0), 0) / 64.0;
				const double a4 = _mm_extract_epi16(_mm256_extracti128_si256(row_avx, 1), 0) / 64.0;

				return category(a1, a2, a3, a4);
			}
#endif
			const auto halfW = image.width() / 2;
			const auto halfH = image.height() / 2;
			const double a1 = ImageStatistics::mean(image.slice(0, 0, halfW, halfH));
			const double a2 = ImageStatistics::mean(image.slice(halfW, 0, halfW, halfH));
			const double a3 = ImageStatistics::mean(image.slice(0, halfH, halfW, halfH));
			const double a4 = ImageStatistics::mean(image.slice(halfW, halfH, halfW, halfH));
			return category(a1, a2, a3, a4);
		}
		static int category(const double a1, const double a2, const double a3, const double a4) noexcept {
			const bool a1a2 = a1 > a2;
			const bool a1a3 = a1 > a3;
			const bool a1a4 = a1 > a4;
			const bool a2a1 = a2 > a1;
			const bool a2a3 = a2 > a3;
			const bool a2a4 = a2 > a4;
			const bool a3a1 = a3 > a1;
			const bool a3a2 = a3 > a2;
			const bool a3a4 = a3 > a4;
			const bool a4a1 = a4 > a1;
			const bool a4a2 = a4 > a2;
			const bool a4a3 = a4 > a3;

			if (a1a2 && a2a3 && a3a4) return 0;
			if (a3a1 && a1a4 && a4a2) return 0;
			if (a4a3 && a3a2 && a2a1) return 0;
			if (a2a4 && a4a1 && a1a3) return 0;

			if (a1a3 && a3a2 && a2a4) return 1;
			if (a2a1 && a1a4 && a4a3) return 1;
			if (a4a2 && a2a3 && a3a1) return 1;
			if (a3a4 && a4a1 && a1a2) return 1;

			if (a1a4 && a4a3 && a3a2) return 2;
			if (a4a1 && a1a2 && a2a3) return 2;
			if (a3a2 && a2a4 && a4a1) return 2;
			if (a2a3 && a3a1 && a1a4) return 2;

			if (a1a2 && a2a4 && a4a3) return 3;
			if (a3a1 && a1a2 && a2a4) return 3;
			if (a4a3 && a3a1 && a1a2) return 3;
			if (a2a4 && a4a3 && a3a1) return 3;

			if (a2a1 && a1a3 && a3a4) return 4;
			if (a1a3 && a3a4 && a4a2) return 4;
			if (a3a4 && a4a2 && a2a1) return 4;
			if (a4a2 && a2a1 && a1a3) return 4;

			if (a1a4 && a4a2 && a2a3) return 5;
			if (a4a1 && a1a3 && a3a4) return 5;
			if (a2a3 && a3a4 && a4a1) return 5;
			if (a3a2 && a2a1 && a1a4) return 5;

			return -1;
		}
	};
}

#endif // CLASSIFIER_H
