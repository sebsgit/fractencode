#pragma once

namespace Frac {

	struct encode_parameters_t {
		int sourceGridSize = 16;
		int targetGridSize = 4;
		int latticeSize = 2;
		double rmsThreshold = 0.0;
		double sMax = -1.0;
		bool nogpu = false;
		bool nocpu = false;
	};

}
