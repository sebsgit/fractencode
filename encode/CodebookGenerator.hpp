#pragma once

#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <set>

namespace Frac {
	class UniqueIndexGenerator {
	public:
		explicit UniqueIndexGenerator(size_t maxIndex, std::random_device& device)
			: _generator(device())
			, _distribution(0, maxIndex)
		{

		}
		void reset() {
			this->_generated.clear();
		}
		size_t countGenerated() const noexcept {
			return this->_generated.size();
		}
		size_t next() {
			if (this->_generated.size() == this->_distribution.b())
				return std::numeric_limits<size_t>::max();
			auto result = this->_distribution(this->_generator);
			while (this->_generated.find(result) != this->_generated.end()) {
				result = this->_distribution(this->_generator);
			}
			this->_generated.insert(result);
			return result;
		}
	private:
		std::set<size_t> _generated;
		std::mt19937 _generator;
		std::uniform_int_distribution<size_t> _distribution;
	};

	namespace ClusterSelector {
		template <typename ItemIterator>
		class FullRange {
		public:
			FullRange(ItemIterator start, ItemIterator end) noexcept
				: _current(start)
				, _end(end)
			{
			}
			bool hasNext() const noexcept {
				return this->_current != this->_end;
			}
			auto next() noexcept {
				return this->_current++;
			}

		private:
			ItemIterator _current;
			const ItemIterator _end;
		};

		template <typename ItemIterator, size_t limit>
		struct LimitRange {
		public:
			LimitRange(ItemIterator start, ItemIterator end) noexcept
				: _generator(std::distance(start, end) - 1, this->_rd)
				, _start(start)
				, _maxCount(std::distance(start, end) - 1)
			{
			}
			bool hasNext() const noexcept {
				return this->_generator.countGenerated() < this->_maxCount && this->_generator.countGenerated() < limit;
			}
			auto next() noexcept {
				return this->_start + this->_generator.next();
			}
		private:
			std::random_device _rd;
			UniqueIndexGenerator _generator;
			const ItemIterator _start;
			const size_t _maxCount;
		};
	};

	template <typename Item, typename ItemIterator, typename ClusterSelectionAlgorithm = ClusterSelector::FullRange<ItemIterator>>
	auto generateCodebook(ItemIterator start,
		ItemIterator end,
		size_t n,
		float epsilon,
		const std::function<float(const Item&, const Item&)>& distance,
		int maxSteps = 200)
	{
		std::vector<Item> result;
		result.reserve(n);

		std::random_device rd;

		UniqueIndexGenerator indexGenerator(std::distance(start, end) - 1, rd);

		auto getClusterIndex = [&distance, n](const auto& codebook, Item a) {
			size_t ind = 0;
			auto currentDist = distance(codebook[0], a);
			for (size_t i = 1; i < n; ++i) {
				auto d = distance(codebook[i], a);
				if (d < currentDist) {
					ind = i;
					currentDist = d;
				}
			}
			return ind;
		};

		// initial codebook
		for (size_t i = 0; i < n; ++i) {
			result.push_back(*(start + indexGenerator.next()));
		}

		float lastChange = epsilon * 2;
		while (lastChange > epsilon && maxSteps > 0) {
			--maxSteps;
			std::vector<std::vector<size_t>> clusters;
			clusters.resize(n);

			// clusterize input
			ClusterSelectionAlgorithm clusterizer(start, end);
			while (clusterizer.hasNext()) {
				auto it = clusterizer.next();
				size_t index = getClusterIndex(result, *it);
				clusters[index].push_back(std::distance(start, it));
			}
			// new set of codewords
			std::vector<Item> newResult;
			newResult.reserve(n);
			size_t i = 0;
			for (const auto & c : clusters) {
				if (c.empty()) {
					newResult.push_back(result[i]);
				}
				else {
					Item newCodeword = {};
					for (auto index : c) {
						newCodeword += *(start + index);
					}
					newResult.push_back(newCodeword / c.size());
				}
				++i;
			}

			float maxChange = 0.0f;
			i = 0;
			for (auto & v : newResult) {
				auto d = distance(v, result[i]);
				if (maxChange < d)
					maxChange = d;
				++i;
			}

			lastChange = maxChange;
			result = std::move(newResult);
		}

		return result;
	}
} // namespace 
