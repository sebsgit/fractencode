#ifndef TRANSFORMMATCHER_H
#define TRANSFORMMATCHER_H

#include "image.h"
#include "transform.h"
#include "metrics.h"
#include "partition.h"
#include "datatypes.h"
#include <iostream>

#ifdef FRAC_WITH_AVX
#include "sse_utils.h"
#include <immintrin.h>
#endif

namespace Frac {
	static const int __map_lookup[8][8] = {
		/*ID*/{ 1, 0, 0, 0,  0, 1, 0, 0 },
		/*90*/{ 0, 1, 0, 0,  -1, 0, 1, 0 },
		/*180*/{ -1, 0, 1, 0,  0, -1, 0, 1 },
		/*270*/{ 0, -1, 0, 1,  1, 0, 0, 0 },
		/*flip*/{ 1, 0, 0, 0,   0, -1, 0, 1 },
		/*fl 90*/{ 0, 1, 0, 0,   1, 0, 0, 0 },
		/*fl 180*/{ -1, 0, 1, 0,  0, 1, 0, 0 },
		/*fl 270*/{ 0, -1, 0, 1, -1, 0, 1, 0 }
	};

class TransformMatcher {
public:
	TransformMatcher(const Metric& metric, const double rmsThreshold, const double sMax)
		:_metric(metric)
		,_rmsThreshold(rmsThreshold)
		,_sMax(sMax)
	{

	}
#ifdef FRAC_WITH_AVX
	transform_score_t match_sse_2x2(const PartitionItemPtr& a, const PartitionItemPtr& b) const {
		transform_score_t result;
		Transform t(Transform::Id);

		const auto stride_b = b->image().stride();
		const __m256i offset_01_avx = _mm256_set_epi16(1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0);
		const __m256i b_stride_avx = frac_m256_interleave2_epi16(stride_b, 1);
		const double N = 4.0;
		const Image::Pixel* source_b = b->image().data()->get();
		const __m256i w_1_avx = _mm256_set1_epi16(b->width() - 1);
		const __m256i h_1_avx = _mm256_set1_epi16(b->height() - 1);
		
		ALIGN_SPEC uint16_t top_coords_avx_store[16] ALIGN_ATTR;
		ALIGN_SPEC uint16_t bottom_coords_avx_store[16] ALIGN_ATTR;

		do {
			transform_score_t candidate;
			candidate.distance = _metric.distance(a->image(), b->presampled(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const __m256i map_lookup_4_0_avx = frac_m256_interleave2_epi16(__map_lookup[t.type()][4], __map_lookup[t.type()][0]);
				const __m256i map_lookup_5_1_avx = frac_m256_interleave2_epi16(__map_lookup[t.type()][5], __map_lookup[t.type()][1]);
				const __m256i map_lookup_6_2_avx = frac_m256_interleave2_epi16(__map_lookup[t.type()][6], __map_lookup[t.type()][2]);
				const __m256i map_lookup_7_3_avx = frac_m256_interleave2_epi16(__map_lookup[t.type()][7], __map_lookup[t.type()][3]);
				
				const __m256i wh_offset_avx = _mm256_add_epi16( _mm256_mullo_epi16(map_lookup_6_2_avx, w_1_avx), _mm256_mullo_epi16(map_lookup_7_3_avx, h_1_avx) );
#ifdef _MSC_VER
				const __m256i ys_avx = _mm256_set_m128i(_mm_set1_epi16(b->image().height() / 2), _mm_setzero_si128());
#else
				const __m256i ys_avx = _mm256_set_epi16(b->image().height() / 2, b->image().height() / 2, b->image().height() / 2, b->image().height() / 2,
					b->image().height() / 2, b->image().height() / 2, b->image().height() / 2, b->image().height() / 2,
					0, 0, 0, 0, 0, 0, 0, 0);
#endif
				__m256i y_wh_offset_avx = _mm256_mullo_epi16(ys_avx, map_lookup_5_1_avx);
				y_wh_offset_avx = _mm256_add_epi16(y_wh_offset_avx, wh_offset_avx);
					
				const auto xs_0 = 0;
				const auto xs_1 = (b->image().width()) / 2;

				__m256i xs_avx = _mm256_add_epi16(frac_m256_interleave4_epi16(xs_1, xs_1, xs_0, xs_0), offset_01_avx);
				xs_avx = _mm256_mullo_epi16(xs_avx, map_lookup_4_0_avx);

				__m256i top_coords_avx = _mm256_add_epi16(xs_avx, y_wh_offset_avx);
				__m256i bottom_coords_avx = _mm256_add_epi16(xs_avx, _mm256_add_epi16(y_wh_offset_avx, map_lookup_5_1_avx));
				top_coords_avx = _mm256_mullo_epi16(top_coords_avx, b_stride_avx);
				bottom_coords_avx = _mm256_mullo_epi16(bottom_coords_avx, b_stride_avx);

				xs_avx = _mm256_shufflehi_epi16(top_coords_avx, _MM_SHUFFLE(2, 3, 0, 1));
				xs_avx = _mm256_shufflelo_epi16(xs_avx, _MM_SHUFFLE(2, 3, 0, 1));
				top_coords_avx = _mm256_add_epi16(top_coords_avx, xs_avx);
				xs_avx = _mm256_shufflehi_epi16(bottom_coords_avx, _MM_SHUFFLE(2, 3, 0, 1));
				xs_avx = _mm256_shufflelo_epi16(xs_avx, _MM_SHUFFLE(2, 3, 0, 1));
				bottom_coords_avx = _mm256_add_epi16(bottom_coords_avx, xs_avx);

				_mm256_store_si256((__m256i*)top_coords_avx_store, top_coords_avx);
				_mm256_store_si256((__m256i*)bottom_coords_avx_store, bottom_coords_avx);

				const int total_0 = (int)source_b[top_coords_avx_store[0]]
				+ (int)source_b[top_coords_avx_store[4]]
				+ (int)source_b[bottom_coords_avx_store[0]]
				+ (int)source_b[bottom_coords_avx_store[4]];
				const int total_1 = (int)source_b[top_coords_avx_store[2]]
				+ (int)source_b[top_coords_avx_store[6]]
				+ (int)source_b[bottom_coords_avx_store[2]]
				+ (int)source_b[bottom_coords_avx_store[6]];

				const int total_2 = (int)source_b[top_coords_avx_store[0 + 8]]
				+ (int)source_b[top_coords_avx_store[4 + 8]]
				+ (int)source_b[bottom_coords_avx_store[0 + 8]]
				+ (int)source_b[bottom_coords_avx_store[4 + 8]];
				const int total_3 = (int)source_b[top_coords_avx_store[2 + 8]]
				+ (int)source_b[top_coords_avx_store[6 + 8]]
				+ (int)source_b[bottom_coords_avx_store[2 + 8]]
				+ (int)source_b[bottom_coords_avx_store[6 + 8]];

				const double valB_0 = convert<double>(total_0 / 4);
				const double valB_1 = convert<double>(total_1 / 4);
				const double valB_2 = convert<double>(total_2 / 4);
				const double valB_3 = convert<double>(total_3 / 4);

				const double valA_0 = convert<double>(a->image().data()->get()[0]);
				const double valA_1 = convert<double>(a->image().data()->get()[1]);
				const double valA_2 = convert<double>(a->image().data()->get()[0 + a->image().stride()]);
				const double valA_3 = convert<double>(a->image().data()->get()[1 + a->image().stride()]);

				const auto sumA = valA_0 + valA_1 + valA_2 + valA_3;
				const auto sumB = valB_0 + valB_1 + valB_2 + valB_3;
				const auto sumA2 = valA_0 * valA_0 + valA_1 * valA_1 + valA_2 * valA_2 + valA_3 * valA_3;
				const auto sumAB = valA_0 * valB_0 + valA_1 * valB_1 + valA_2 * valB_2 + valA_3 * valB_3;
				const double tmp = (N * sumA2 - (sumA - 1) * sumA);
				const double s = this->truncateSMax(fabs(tmp) < 0.00001 ? 0.0 : (N * sumAB - sumA * sumB) / tmp);
				const double o = (sumB - s * sumA) / N;
				candidate.contrast = s;
				candidate.brightness = o;
				result = candidate;
			}
			if (this->checkDistance(result.distance))
				break;
		} while (t.next() != Transform::Id);
		return result;
	}

	//TODO
	transform_score_t match_sse_4x4(const PartitionItemPtr& a, const PartitionItemPtr& b) const {
		transform_score_t result;
		Transform t(Transform::Id);
		do {
			transform_score_t candidate;
			candidate.distance = _metric.distance(a->image(), b->presampled(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = (double)(a->image().width()) * a->image().height();
				double sumA = 0.0, sumA2 = 0.0, sumB = 0.0, sumAB = 0.0;
				const Image::Pixel* source_b = b->image().data()->get();
				const auto stride_b = b->image().stride();
				const auto width_offset = __map_lookup[t.type()][2] * (b->width() - 1) + __map_lookup[t.type()][3] * (b->height() - 1);
				const auto height_offset = __map_lookup[t.type()][6] * (b->width() - 1) + __map_lookup[t.type()][7] * (b->height() - 1);
				for (uint32_t y = 0; y<a->image().height(); ++y) {
					const auto srcY = (y * b->image().height()) / a->image().height();
					auto ys = srcY;
					if (ys == b->height() - 1)
						--ys;
					const auto y_width_offset = __map_lookup[t.type()][1] * ys + width_offset;
					const auto y_height_offset = __map_lookup[t.type()][5] * ys + height_offset;
					const auto y_width_offset_1 = __map_lookup[t.type()][1] * (ys + 1) + width_offset;
					const auto y_height_offset_1 = __map_lookup[t.type()][5] * (ys + 1) + height_offset;

					const auto map_x0 = __map_lookup[t.type()][0];
					const auto map_x1 = __map_lookup[t.type()][4];
					const auto y_off = y * a->image().stride();

					for (uint32_t x = 0; x<a->image().width(); ++x) {

						// mm_loadu_128(a->data() + y_off + x)
						const double valA = convert<double>(a->image().data()->get()[x + y_off]);

						// A = [ x, x+1, x+2, ..., x+7 ] = [ x, x, x, ..., x ] + [0, 1, 2, 3, 4, ..., 7]
						// B = [ b / a, ..., b / a ]
						// A * B
						auto srcX = (x * b->image().width()) / a->image().width();

						// A = [0, ..., 1]
						// mask = [0, ..., x+8 == b->w]
						// r = r - A * mask
						auto xs = srcX;
						if (xs == b->width() - 1)
							--xs;

						// M0 = [ map_x0, ..., map_x0 ]
						// M1 = [map_x1, ..., map_x1]
						// _1 = [1, ..., 1]

						// XS_X = M0 * r
						const auto xs_x = map_x0 * xs;
						// XS_Y = M1 * r
						const auto xs_y = map_x1 * xs;
						// XS_X_1 = M0 * (r + _1)
						const auto xs_x_1 = map_x0 * (xs + 1);
						// XS_Y_1 = M1 * (r + 1)
						const auto xs_y_1 = map_x1 * (xs + 1);

						// TL_X = XSX + [Y_W_OFF]
						// TL_Y = (XSY + [Y_H_OFF]) * B_STRIDE
						auto tl = Point2d<uint32_t>(xs_x + y_width_offset, xs_y + y_height_offset);
						auto tr = Point2d<uint32_t>(xs_x_1 + y_width_offset, xs_y_1 + y_height_offset);
						auto bl = Point2d<uint32_t>(xs_x + y_width_offset_1, xs_y + y_height_offset_1);
						auto br = Point2d<uint32_t>(xs_x_1 + y_width_offset_1, xs_y_1 + y_height_offset_1);

						const int total = (int)source_b[tl.x() + tl.y() * stride_b]
							+ (int)source_b[tr.x() + tr.y() * stride_b]
							+ (int)source_b[bl.x() + bl.y() * stride_b]
							+ (int)source_b[br.x() + br.y() * stride_b];
						const Image::Pixel sample_b = (total / 4);

						const double valB = convert<double>(sample_b);
						sumA += valA;
						sumB += valB;
						sumA2 += valA * valA;
						sumAB += valA * valB;
					}
				}
				const double tmp = (N * sumA2 - (sumA - 1) * sumA);
				const double s = this->truncateSMax(fabs(tmp) < 0.00001 ? 0.0 : (N * sumAB - sumA * sumB) / tmp);
				const double o = (sumB - s * sumA) / N;
				candidate.contrast = s;
				candidate.brightness = o;
				result = candidate;
			}
			if (this->checkDistance(result.distance))
				break;
		} while (t.next() != Transform::Id);
		return result;
	}


	transform_score_t match_sse_8x8(const PartitionItemPtr& a, const PartitionItemPtr& b) const {
		transform_score_t result;
		static const int __map_lookup[8][8] = {
			/*ID*/{ 1, 0, 0, 0,  0, 1, 0, 0 },
			/*90*/{ 0, 1, 0, 0,  -1, 0, 1, 0 },
			/*180*/{ -1, 0, 1, 0,  0, -1, 0, 1 },
			/*270*/{ 0, -1, 0, 1,  1, 0, 0, 0 },
			/*flip*/{ 1, 0, 0, 0,   0, -1, 0, 1 },
			/*fl 90*/{ 0, 1, 0, 0,   1, 0, 0, 0 },
			/*fl 180*/{ -1, 0, 1, 0,  0, 1, 0, 0 },
			/*fl 270*/{ 0, -1, 0, 1, -1, 0, 1, 0 }
		};
		Transform t(Transform::Id);
		__m128i ab_ratio_sse = _mm_set1_epi16(b->width() / a->width());
		do {
			transform_score_t candidate;
			candidate.distance = _metric.distance(a->image(), b->presampled(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = (double)(a->image().width()) * a->image().height();
				double sumA = 0.0, sumA2 = 0.0, sumB = 0.0, sumAB = 0.0;
				const Image::Pixel* source_b = b->image().data()->get();
				const auto width_offset = __map_lookup[t.type()][2] * (b->width() - 1) + __map_lookup[t.type()][3] * (b->height() - 1);
				const auto height_offset = __map_lookup[t.type()][6] * (b->width() - 1) + __map_lookup[t.type()][7] * (b->height() - 1);

				__m128i sumA_sse = _mm_setzero_si128();
				__m128i x_increase_sse = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
				__m128i map_x0_sse = _mm_set1_epi16(__map_lookup[t.type()][0]);
				__m128i map_x1_sse = _mm_set1_epi16(__map_lookup[t.type()][4]);
				__m128i b_stride_sse = _mm_set1_epi16(b->image().stride());

				ALIGN_SPEC uint16_t tmp_sse[8] ALIGN_ATTR = { 0 };
				ALIGN_SPEC uint16_t tl_store[8] ALIGN_ATTR;
				ALIGN_SPEC uint16_t tr_store[8] ALIGN_ATTR;
				ALIGN_SPEC uint16_t bl_store[8] ALIGN_ATTR;
				ALIGN_SPEC uint16_t br_store[8] ALIGN_ATTR;

				for (uint32_t y = 0; y<a->image().height(); ++y) {
					const auto srcY = (y * b->image().height()) / a->image().height();
					auto ys = srcY;
					if (ys == b->height() - 1)
						--ys;
					const auto y_width_offset = __map_lookup[t.type()][1] * ys + width_offset;
					const auto y_height_offset = __map_lookup[t.type()][5] * ys + height_offset;
					const auto y_width_offset_1 = __map_lookup[t.type()][1] * (ys + 1) + width_offset;
					const auto y_height_offset_1 = __map_lookup[t.type()][5] * (ys + 1) + height_offset;
					const auto y_off = y * a->image().stride();

					__m128i y_w_off_sse = _mm_set1_epi16(y_width_offset);
					__m128i y_h_off_sse = _mm_set1_epi16(y_height_offset);
					__m128i y_w_off_sse_1 = _mm_set1_epi16(y_width_offset_1);
					__m128i y_h_off_sse_1 = _mm_set1_epi16(y_height_offset_1);

					for (uint32_t x = 0; x<a->image().width(); x += 8) {
						auto src_data = (a->image().data()->get() + x + y_off);
						__m128i x_sse = _mm_set_epi64x(0, *(uint64_t*)src_data);
						x_sse = _mm_unpacklo_epi8(x_sse, _mm_setzero_si128());
						sumA_sse = _mm_add_epi16(x_sse, sumA_sse);
						x_sse = _mm_mullo_epi16(x_sse, x_sse);
						_mm_store_si128((__m128i*)tmp_sse, x_sse);
						sumA2 += tmp_sse[0] + tmp_sse[1] + tmp_sse[2] + tmp_sse[3] + tmp_sse[4] + tmp_sse[5] + tmp_sse[6] + tmp_sse[7];

						x_sse = _mm_set1_epi16(x);
						x_sse = _mm_add_epi16(x_sse, x_increase_sse);

						__m128i srcX_sse = _mm_mullo_epi16(x_sse, ab_ratio_sse);
						srcX_sse = _mm_sub_epi16(srcX_sse, _mm_set_epi16(x + 8 == b->width() ? 1 : 0, 0, 0, 0, 0, 0, 0, 0));

						__m128i xsx_sse = _mm_mullo_epi16(map_x0_sse, srcX_sse);
						__m128i xsy_sse = _mm_mullo_epi16(map_x1_sse, srcX_sse);
						__m128i xsx_sse_1 = _mm_mullo_epi16(map_x0_sse, _mm_add_epi16(srcX_sse, _mm_set1_epi16(1)));
						__m128i xsy_sse_1 = _mm_mullo_epi16(map_x1_sse, _mm_add_epi16(srcX_sse, _mm_set1_epi16(1)));

						__m128i tl_x = _mm_add_epi16(xsx_sse, y_w_off_sse);
						__m128i tl_y = _mm_add_epi16(xsy_sse, y_h_off_sse);
						tl_y = _mm_mullo_epi16(tl_y, b_stride_sse);
						tl_x = _mm_add_epi16(tl_x, tl_y);

						__m128i tr_x = _mm_add_epi16(xsx_sse_1, y_w_off_sse);
						__m128i tr_y = _mm_add_epi16(xsy_sse_1, y_h_off_sse);
						tr_y = _mm_mullo_epi16(tr_y, b_stride_sse);
						tr_x = _mm_add_epi16(tr_x, tr_y);

						__m128i bl_x = _mm_add_epi16(xsx_sse, y_w_off_sse_1);
						__m128i bl_y = _mm_add_epi16(xsy_sse, y_h_off_sse_1);
						bl_y = _mm_mullo_epi16(bl_y, b_stride_sse);
						bl_x = _mm_add_epi16(bl_x, bl_y);

						__m128i br_x = _mm_add_epi16(xsx_sse_1, y_w_off_sse_1);
						__m128i br_y = _mm_add_epi16(xsy_sse_1, y_h_off_sse_1);
						br_y = _mm_mullo_epi16(br_y, b_stride_sse);
						br_x = _mm_add_epi16(br_x, br_y);

						_mm_store_si128((__m128i*)tl_store, tl_x);
						_mm_store_si128((__m128i*)tr_store, tr_x);
						_mm_store_si128((__m128i*)bl_store, bl_x);
						_mm_store_si128((__m128i*)br_store, br_x);

						//TODO maybe vectorize this
						for (int i = 0; i < 8; ++i) {
							const int total = (int)source_b[tl_store[i]]
								+ (int)source_b[tr_store[i]]
								+ (int)source_b[bl_store[i]]
								+ (int)source_b[br_store[i]];
							sumAB += convert<double>(total / 4) * src_data[i];
							sumB += convert<double>(total / 4);
						}
					}
				}

				_mm_store_si128((__m128i*)tmp_sse, sumA_sse);
				sumA = tmp_sse[0] + tmp_sse[1] + tmp_sse[2] + tmp_sse[3] + tmp_sse[4] + tmp_sse[5] + tmp_sse[6] + tmp_sse[7];

				const double tmp = (N * sumA2 - (sumA - 1) * sumA);
				const double s = this->truncateSMax(fabs(tmp) < 0.00001 ? 0.0 : (N * sumAB - sumA * sumB) / tmp);
				const double o = (sumB - s * sumA) / N;
				candidate.contrast = s;
				candidate.brightness = o;
				result = candidate;
			}
			if (this->checkDistance(result.distance))
				break;
		} while (t.next() != Transform::Id);
		return result;
	}
#endif
	transform_score_t match_default(const PartitionItemPtr& a, const PartitionItemPtr& b) const {
		transform_score_t result;
		Transform t(Transform::Id);
		const SamplerBilinear samplerB(b->image());
		do {
			transform_score_t candidate;
			candidate.distance = _metric.distance(a->image(), b->presampled(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = (double)(a->image().width()) * a->image().height();
				double sumA = ImageStatistics::sum(a->image()), sumA2 = 0.0, sumB = 0.0, sumAB = 0.0;
				for (uint32_t y = 0 ; y<a->image().height() ; ++y) {
					for (uint32_t x = 0 ; x<a->image().width() ; ++x) {
						const auto srcY = (y * b->image().height()) / a->image().height();
						const auto srcX = (x * b->image().width()) / a->image().width();
						const double valA = convert<double>(a->image().data()->get()[x + y * a->image().stride()]);
						const double valB = convert<double>(samplerB(srcX, srcY, t, b->image().size()));
						sumB += valB;
						sumA2 += valA * valA;
						sumAB += valA * valB;
					}
				}
				const double tmp = (N * sumA2 - (sumA - 1) * sumA);
				const double s = this->truncateSMax( fabs(tmp) < 0.00001 ? 0.0 : (N * sumAB - sumA * sumB) / tmp );
				const double o = (sumB - s * sumA) / N;
				candidate.contrast = s;
				candidate.brightness = o;
				result = candidate;
			}
			if (this->checkDistance(result.distance))
				break;
		} while (t.next() != Transform::Id);
		return result;
	}

	inline transform_score_t match(const PartitionItemPtr& a, const PartitionItemPtr& b) const {
#ifdef FRAC_WITH_AVX
		if (a->width() == 2)
			return this->match_sse_2x2(a, b);
		else if (a->width() == 8)
			return this->match_sse_8x8(a, b);
		return this->match_default(a, b);
#else
		return this->match_default(a, b);
#endif
	}
	double truncateSMax(const double s) const noexcept {
		if (_sMax > 0.0)
			return s > _sMax ? _sMax : (s < -_sMax ? -_sMax : s);
		return s;
	}
	bool checkDistance(const double d) const noexcept {
		return d <= _rmsThreshold;
	}
private:
	const Metric& _metric;
	const double _rmsThreshold;
	const double _sMax;
};
}

#endif // TRANSFORMMATCHER_H
