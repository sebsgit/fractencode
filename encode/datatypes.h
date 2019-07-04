#ifndef DATATYPES_H
#define DATATYPES_H

#include "image/transform.h"
#include <vector>

namespace Frac {
	struct transform_score_t {
		double distance = 100000.0;
		double contrast = 0.0;
		double brightness = 0.0;
		Transform::Type transform = Transform::Id;
	};
	struct item_match_t {
		transform_score_t score;
		uint32_t x = 0;
		uint32_t y = 0;
		Size32u sourceItemSize;
	};
	struct encode_item_t {
		uint32_t x, y, w, h;
		item_match_t match;
	};
	struct grid_encode_data_t {
		std::vector<encode_item_t> encoded;
	};
}

#endif // DATATYPES_H
