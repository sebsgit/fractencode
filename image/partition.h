#ifndef PARTITION_H
#define PARTITION_H

#include "image/image.h"
#include "encode/datatypes.h"
#include <vector>
#include <ostream>

namespace Frac {
	class TransformMatcher;
	class ImageClassifier;

	class Partition;
	using PartitionPtr = std::shared_ptr<Partition>;

	class GridItem {
	public:
		GridItem(const Image& source, const uint32_t x, const uint32_t y, const Size32u& s, Image&& presampled = Image())
			:_pos(x, y)
			,_sourceSize(s)
			,_image(source.slice(x, y, s.x(), s.y()))
			,_presampled(std::move(presampled))
		{
		}
		GridItem(const Image& source, const uint32_t x, const uint32_t y, Image&& presampled = Image())
			:_pos(x, y)
			,_sourceSize(source.size())
			,_image(source)
			,_presampled(std::move(presampled))
		{
		}
		~GridItem() {}
		Image& image() noexcept {
			return _image;
		}
		Image image() const noexcept {
			return _image;
		}
		const Image& presampled() const noexcept {
			return _presampled.empty() ? _image : _presampled;
		}
		uint32_t x() const {
			return _pos.x();
		}
		uint32_t y() const {
			return _pos.y();
		}
		const Point2du pos() const noexcept {
			return _pos;
		}
		uint32_t width() const noexcept {
			return _image.width();
		}
		uint32_t height() const noexcept {
			return _image.height();
		}
		const Size32u size() const noexcept {
			return _image.size();
		}
		const Size32u sourceSize() const noexcept {
			return _sourceSize;
		}
	private:
		const Point2du _pos;
		const Size32u _sourceSize;
		Image _image;
		const Image _presampled;
	};

	using PartitionItemPtr = std::shared_ptr<GridItem>;

	class Partition {
	public:
		Partition() {

		}
		virtual ~Partition() {

		}
		const std::vector<PartitionItemPtr>::const_iterator begin() const {
			return _data.begin();
		}
		const std::vector<PartitionItemPtr>::const_iterator end() const {
			return _data.end();
		}
		std::vector<PartitionItemPtr>::iterator begin() {
			return _data.begin();
		}
		std::vector<PartitionItemPtr>::iterator end() {
			return _data.end();
		}
		size_t size() const {
			return _data.size();
		}
		void push_back(const PartitionItemPtr& p) {
			this->_data.push_back(p);
		}
		void merge(const PartitionPtr& other) {
			_data.insert(end(), other->begin(), other->end());
		}

	protected:
		std::vector<PartitionItemPtr> _data;
	};

	class PartitionCreator {
	public:
		virtual ~PartitionCreator() {}
		virtual PartitionPtr create(const Image&) const = 0;
	};

}

std::ostream& operator << (std::ostream& out, const Frac::Size32u& s);

#endif // PARTITION_H
