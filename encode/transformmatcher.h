#ifndef TRANSFORMMATCHER_H
#define TRANSFORMMATCHER_H

#include "image.h"
#include "transform.h"
#include "metrics.h"
#include "partition.h"
#include "datatypes.h"
#include <iostream>

#ifdef FRAC_WITH_AVX
#include "immintrin.h"
#endif

namespace Frac {
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
		transform_score_t result;

		Transform t(Transform::Id);
		do {
			transform_score_t candidate;
			candidate.distance = _metric.distance(a->image(), b->image(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = 4.0;
				double sumA = 0.0, sumA2 = 0.0, sumB = 0.0, sumB2 = 0.0, sumAB = 0.0;
				const Image::Pixel* source_b = b->image().data()->get();
				const auto stride_b = b->image().stride();
				const auto width_offset = __map_lookup[t.type()][2] * (b->width() - 1) + __map_lookup[t.type()][3] * (b->height() - 1);
				const auto height_offset = __map_lookup[t.type()][6] * (b->width() - 1) + __map_lookup[t.type()][7] * (b->height() - 1);
				for (uint32_t y = 0; y<2; ++y) {
					const auto ys = (y * b->image().height()) / 2;
					const auto y_width_offset = __map_lookup[t.type()][1] * ys + width_offset;
					const auto y_height_offset = __map_lookup[t.type()][5] * ys + height_offset;
					const auto y_width_offset_1 = __map_lookup[t.type()][1] * (ys + 1) + width_offset;
					const auto y_height_offset_1 = __map_lookup[t.type()][5] * (ys + 1) + height_offset;

					const auto map_x0 = __map_lookup[t.type()][0];
					const auto map_x1 = __map_lookup[t.type()][4];
					const auto y_off = y * a->image().stride();

					const double valA_0 = convert<double>(a->image().data()->get()[0 + y_off]);
					const double valA_1 = convert<double>(a->image().data()->get()[1 + y_off]);

					const auto xs_0 = 0;
					const auto xs_1 = (b->image().width()) / 2;
					const auto xs_x_0 = map_x0 * xs_0;
					const auto xs_y_0 = map_x1 * xs_0;
					const auto xs_x_1 = map_x0 * xs_1;
					const auto xs_y_1 = map_x1 * xs_1;
					const auto xs_x_1_0 = map_x0 * (xs_0 + 1);
					const auto xs_y_1_0 = map_x1 * (xs_0 + 1);
					const auto xs_x_1_1 = map_x0 * (xs_1 + 1);
					const auto xs_y_1_1 = map_x1 * (xs_1 + 1);

					int tl_0_x = xs_x_0 + y_width_offset;
					int tl_0_y = xs_y_0 + y_height_offset;
					int tr_0_x = xs_x_1_0 + y_width_offset;
					int tr_0_y = xs_y_1_0 + y_height_offset;
					int bl_0_x = 0 + y_width_offset_1;
					int bl_0_y = xs_y_0 + y_height_offset_1;
					int br_0_x = xs_x_1_0 + y_width_offset_1;
					int br_0_y = xs_y_1_0 + y_height_offset_1;
					auto tl_1 = Point2d<uint32_t>(xs_x_1 + y_width_offset, xs_y_1 + y_height_offset);
					auto tr_1 = Point2d<uint32_t>(xs_x_1_1 + y_width_offset, xs_y_1_1 + y_height_offset);
					auto bl_1 = Point2d<uint32_t>(xs_x_1 + y_width_offset_1, xs_y_1 + y_height_offset_1);
					auto br_1 = Point2d<uint32_t>(xs_x_1_1 + y_width_offset_1, xs_y_1_1 + y_height_offset_1);
					
					const int total_0 = (int)source_b[tl_0_x + tl_0_y * stride_b]
						+ (int)source_b[tr_0_x + tr_0_y * stride_b]
						+ (int)source_b[bl_0_x + bl_0_y * stride_b]
						+ (int)source_b[br_0_x + br_0_y * stride_b];
					const int total_1 = (int)source_b[tl_1.x() + tl_1.y() * stride_b]
						+ (int)source_b[tr_1.x() + tr_1.y() * stride_b]
						+ (int)source_b[bl_1.x() + bl_1.y() * stride_b]
						+ (int)source_b[br_1.x() + br_1.y() * stride_b];

					const double valB_0 = convert<double>(total_0 / 4);
					const double valB_1 = convert<double>(total_1 / 4);
					
					sumA += valA_0;
					sumB += valB_0;
					sumA2 += valA_0 * valA_0;
					sumB2 += valB_0 * valB_0;
					sumAB += valA_0 * valB_0;
					
					sumA += valA_1;
					sumB += valB_1;
					sumA2 += valA_1 * valA_1;
					sumB2 += valB_1 * valB_1;
					sumAB += valA_1 * valB_1;
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

	//TODO
	transform_score_t match_sse_4x4(const PartitionItemPtr& a, const PartitionItemPtr& b) const {
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
		transform_score_t result;

		Transform t(Transform::Id);
		do {
			transform_score_t candidate;
			candidate.distance = _metric.distance(a->image(), b->image(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = (double)(a->image().width()) * a->image().height();
				double sumA = 0.0, sumA2 = 0.0, sumB = 0.0, sumB2 = 0.0, sumAB = 0.0;
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
						sumB2 += valB * valB;
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
			candidate.distance = _metric.distance(a->image(), b->image(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = (double)(a->image().width()) * a->image().height();
				double sumA = 0.0, sumA2 = 0.0, sumB = 0.0, sumB2 = 0.0, sumAB = 0.0;
				const Image::Pixel* source_b = b->image().data()->get();
				const auto stride_b = b->image().stride();
				const auto width_offset = __map_lookup[t.type()][2] * (b->width() - 1) + __map_lookup[t.type()][3] * (b->height() - 1);
				const auto height_offset = __map_lookup[t.type()][6] * (b->width() - 1) + __map_lookup[t.type()][7] * (b->height() - 1);

				__m128i sumA_sse = _mm_setzero_si128();
				__m128i x_increase_sse = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
				__m128i map_x0_sse = _mm_set1_epi16(__map_lookup[t.type()][0]);
				__m128i map_x1_sse = _mm_set1_epi16(__map_lookup[t.type()][4]);
				__m128i b_stride_sse = _mm_set1_epi16(b->image().stride());

				uint16_t tmp_sse[8] = { 0 };
				uint16_t tl_store[8];
				uint16_t tr_store[8];
				uint16_t bl_store[8];
				uint16_t br_store[8];

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

					__m128i y_w_off_sse = _mm_set1_epi16(y_width_offset);
					__m128i y_h_off_sse = _mm_set1_epi16(y_height_offset);
					__m128i y_w_off_sse_1 = _mm_set1_epi16(y_width_offset_1);
					__m128i y_h_off_sse_1 = _mm_set1_epi16(y_height_offset_1);

					for (uint32_t x = 0; x<a->image().width(); x += 8) {
						auto src_data = (a->image().data()->get() + x + y_off);
						__m128i x_sse = _mm_cvtsi64_si128(*(uint64_t*)src_data);
						x_sse = _mm_unpacklo_epi8(x_sse, _mm_setzero_si128());
						sumA_sse = _mm_add_epi16(x_sse, sumA_sse);
						x_sse = _mm_mullo_epi16(x_sse, x_sse);
						_mm_storeu_si128((__m128i*)tmp_sse, x_sse);
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

						_mm_storeu_si128((__m128i*)tl_store, tl_x);
						_mm_storeu_si128((__m128i*)tr_store, tr_x);
						_mm_storeu_si128((__m128i*)bl_store, bl_x);
						_mm_storeu_si128((__m128i*)br_store, br_x);

						//TODO maybe vectorize this
						for (int i = 0; i < 8; ++i) {
							const int total = (int)source_b[tl_store[i]]
								+ (int)source_b[tr_store[i]]
								+ (int)source_b[bl_store[i]]
								+ (int)source_b[br_store[i]];
							sumAB += convert<double>(total / 4) * src_data[i];
							sumB2 += convert<double>(total / 4) * convert<double>(total / 4);
							sumB += convert<double>(total / 4);
						}
					}
				}

				_mm_storeu_si128((__m128i*)tmp_sse, sumA_sse);
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
			candidate.distance = _metric.distance(a->image(), b->image(), t);
			candidate.transform = t.type();
			if (candidate.distance <= result.distance) {
				const double N = (double)(a->image().width()) * a->image().height();
				double sumA = ImageStatistics::sum(a->image()), sumA2 = 0.0, sumB = 0.0, sumB2 = 0.0, sumAB = 0.0;
				for (uint32_t y = 0 ; y<a->image().height() ; ++y) {
					for (uint32_t x = 0 ; x<a->image().width() ; ++x) {
						const auto srcY = (y * b->image().height()) / a->image().height();
						const auto srcX = (x * b->image().width()) / a->image().width();
						const double valA = convert<double>(a->image().data()->get()[x + y * a->image().stride()]);
						const double valB = convert<double>(samplerB(srcX, srcY, t, b->image().size()));
						sumB += valB;
						sumA2 += valA * valA;
						sumB2 += valB * valB;
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
