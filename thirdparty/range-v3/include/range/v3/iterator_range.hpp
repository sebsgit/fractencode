/// \file
// Range v3 library
//
//  Copyright Eric Niebler 2013-present
//
//  Use, modification and distribution is subject to the
//  Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// Project home: https://github.com/ericniebler/range-v3
//

#ifndef RANGES_V3_ITERATOR_RANGE_HPP
#define RANGES_V3_ITERATOR_RANGE_HPP

#include <type_traits>
#include <utility>

#include <meta/meta.hpp>

#include <concepts/concepts.hpp>

#include <range/v3/range_fwd.hpp>

#include <range/v3/iterator/concepts.hpp>
#include <range/v3/iterator/operations.hpp>
#include <range/v3/iterator/unreachable_sentinel.hpp>
#include <range/v3/utility/static_const.hpp>
#include <range/v3/view/interface.hpp>

RANGES_DEPRECATED_HEADER(
    "This header is deprecated. Please switch to subrange in "
    "<range/v3/view/subrange.hpp>.")

namespace ranges
{
    /// \addtogroup group-views
    /// @{

    /// \cond
    namespace _iterator_range_
    {
        struct adl_hook_
        {};

        // A temporary iterator_range can be safely passed to ranges::begin and
        // ranges::end.
        template<class I, class S>
        constexpr I begin(iterator_range<I, S> && r) noexcept(
            std::is_nothrow_copy_constructible<I>::value)
        {
            return r.begin();
        }
        template<class I, class S>
        constexpr I begin(iterator_range<I, S> const && r) noexcept(
            std::is_nothrow_copy_constructible<I>::value)
        {
            return r.begin();
        }
        template<class I, class S>
        constexpr S end(iterator_range<I, S> && r) noexcept(
            std::is_nothrow_copy_constructible<S>::value)
        {
            return r.end();
        }
        template<class I, class S>
        constexpr S end(iterator_range<I, S> const && r) noexcept(
            std::is_nothrow_copy_constructible<S>::value)
        {
            return r.end();
        }

        // A temporary sized_iterator_range can be safely passed to ranges::begin and
        // ranges::end.
        template<class I, class S>
        constexpr I begin(sized_iterator_range<I, S> && r) noexcept(
            std::is_nothrow_copy_constructible<I>::value)
        {
            return r.begin();
        }
        template<class I, class S>
        constexpr I begin(sized_iterator_range<I, S> const && r) noexcept(
            std::is_nothrow_copy_constructible<I>::value)
        {
            return r.begin();
        }
        template<class I, class S>
        constexpr S end(sized_iterator_range<I, S> && r) noexcept(
            std::is_nothrow_copy_constructible<S>::value)
        {
            return r.end();
        }
        template<class I, class S>
        constexpr S end(sized_iterator_range<I, S> const && r) noexcept(
            std::is_nothrow_copy_constructible<S>::value)
        {
            return r.end();
        }
    } // namespace _iterator_range_
    /// \endcond

    template<typename I, typename S /*= I*/>
    struct RANGES_EMPTY_BASES iterator_range
      : view_interface<iterator_range<I, S>,
                       Same<S, unreachable_sentinel_t> ? infinite : unknown>
      , compressed_pair<I, S>
      , _iterator_range_::adl_hook_
    {
    private:
        template<typename, typename>
        friend struct iterator_range;
        template<typename, typename>
        friend struct sized_iterator_range;
        compressed_pair<I, S> & base() noexcept
        {
            return *this;
        }
        compressed_pair<I, S> const & base() const noexcept
        {
            return *this;
        }
        using compressed_pair<I, S>::first;
        using compressed_pair<I, S>::second;

    public:
        using iterator = I;
        using sentinel = S;
        /// \cond
        using const_iterator = I; // Mostly to avoid spurious errors in Boost.Range
        /// \endcond

        constexpr I & begin() &
        {
            return this->first();
        }
        constexpr I const & begin() const &
        {
            return this->first();
        }

        constexpr S & end() &
        {
            return this->second();
        }
        constexpr S const & end() const &
        {
            return this->second();
        }

        iterator_range() = default;
        constexpr iterator_range(I begin, S end)
          : compressed_pair<I, S>{detail::move(begin), detail::move(end)}
        {}
        template<typename X, typename Y>
        constexpr CPP_ctor(iterator_range)(iterator_range<X, Y> rng)( //
            requires Constructible<I, X> && Constructible<S, Y>)
          : compressed_pair<I, S>{detail::move(rng.begin()), detail::move(rng.end())}
        {}
        template<typename X, typename Y>
        explicit constexpr CPP_ctor(iterator_range)(std::pair<X, Y> rng)( //
            requires Constructible<I, X> && Constructible<S, Y>)
          : compressed_pair<I, S>{detail::move(rng.first), detail::move(rng.second)}
        {}
        template<typename X, typename Y>
        auto operator=(iterator_range<X, Y> rng) -> CPP_ret(iterator_range &)( //
            requires Assignable<I &, X> && Assignable<S &, Y>)
        {
            base().first() = std::move(rng.base()).first();
            base().second() = std::move(rng.base()).second();
            return *this;
        }
        CPP_template(typename X, typename Y)(                    //
            requires ConvertibleTo<I, X> && ConvertibleTo<S, Y>) //
            constexpr
            operator std::pair<X, Y>() const
        {
            return {base().first(), base().second()};
        }
        constexpr bool empty() const
        {
            return base().first() == base().second();
        }
    };

    // Like iterator_range, but with a known size. The first and second members
    // are private to prevent inadvertent violations of the class invariant.
    //
    // Class invariant:
    //   distance(begin(), end()) == size()
    //
    template<typename I, typename S /* = I */>
    struct sized_iterator_range
      : view_interface<sized_iterator_range<I, S>, finite>
      , _iterator_range_::adl_hook_
    {
        using size_type = detail::iter_size_t<I>;
        using iterator = I;
        using sentinel = S;
#ifndef RANGES_DOXYGEN_INVOKED
        using const_iterator = I; // Mostly to avoid spurious errors in Boost.Range
#endif

    private:
        template<typename X, typename Y>
        friend struct sized_iterator_range;
        iterator_range<I, S> rng_;
        size_type size_;

    public:
        sized_iterator_range() = default;
        RANGES_NDEBUG_CONSTEXPR sized_iterator_range(I begin, S end, size_type size)
          : rng_{detail::move(begin), detail::move(end)}
          , size_(size)
        {
#ifndef NDEBUG
            RANGES_ASSERT(!(bool)ForwardIterator<I> ||
                          static_cast<size_type>(ranges::distance(rng_)) == size_);
#endif
        }
        template<typename X, typename Y>
        RANGES_NDEBUG_CONSTEXPR CPP_ctor(sized_iterator_range)(std::pair<X, Y> rng,
                                                               size_type size)( //
            requires Constructible<I, X> && Constructible<S, Y>)
          : sized_iterator_range{detail::move(rng).first, detail::move(rng).second, size}
        {}
        template<typename X, typename Y>
        RANGES_NDEBUG_CONSTEXPR CPP_ctor(sized_iterator_range)(iterator_range<X, Y> rng,
                                                               size_type size)( //
            requires Constructible<I, X> && Constructible<S, Y>)
          : sized_iterator_range{detail::move(rng).first(),
                                 detail::move(rng).second,
                                 size}
        {}
        template<typename X, typename Y>
        RANGES_NDEBUG_CONSTEXPR CPP_ctor(sized_iterator_range)(
            sized_iterator_range<X, Y> rng)( //
            requires Constructible<I, X> && Constructible<S, Y>)
          : sized_iterator_range{detail::move(rng).rng_.first(),
                                 detail::move(rng).rng_.second,
                                 rng.size_}
        {}
        template<typename X, typename Y>
        auto operator=(sized_iterator_range<X, Y> rng)
            -> CPP_ret(sized_iterator_range &)( //
                requires Assignable<I &, X> && Assignable<S &, Y>)
        {
            rng_ = detail::move(rng).rng_;
            size_ = rng.size_;
            return *this;
        }
        I begin() const
        {
            return rng_.begin();
        }
        S end() const
        {
            return rng_.end();
        }
        size_type size() const noexcept
        {
            return size_;
        }
        CPP_template(typename X, typename Y)(                    //
            requires ConvertibleTo<I, X> && ConvertibleTo<S, Y>) //
            constexpr
            operator std::pair<X, Y>() const
        {
            return rng_;
        }
        CPP_template(typename X, typename Y)(                    //
            requires ConvertibleTo<I, X> && ConvertibleTo<S, Y>) //
            constexpr
            operator iterator_range<X, Y>() const
        {
            return rng_;
        }
        constexpr operator iterator_range<I, S> const &() const & noexcept
        {
            return rng_;
        }
        // clang-format off
        /// Tuple-like access for `sized_iterator_range`
        CPP_template(std::size_t N)( //
            requires (N < 2))        //
        friend constexpr auto CPP_auto_fun(get)(sized_iterator_range const &p)
        (
            // return ranges::get<N>(p.rng_)
            return ranges::get<N>(p.*&sized_iterator_range::rng_) // makes clang happy
        )
            // clang-format on
            /// \overload
            template<std::size_t N>
            friend constexpr auto get(sized_iterator_range const & p) noexcept
            -> CPP_ret(size_type)( //
                requires (N == 2))
        {
            return p.size();
        }
    };

    struct make_iterator_range_fn
    {
        /// \return `{begin, end}`
        template<typename I, typename S>
        constexpr auto operator()(I begin, S end) const
            -> CPP_ret(iterator_range<I, S>)( //
                requires Sentinel<S, I>)
        {
            return {detail::move(begin), detail::move(end)};
        }

        /// \return `{begin, end, size}`
        template<typename I, typename S>
        constexpr auto operator()(I begin, S end, detail::iter_size_t<I> size) const
            -> CPP_ret(sized_iterator_range<I, S>)( //
                requires Sentinel<S, I>)
        {
            return {detail::move(begin), detail::move(end), size};
        }
    };

    /// \sa `make_iterator_range_fn`
    RANGES_INLINE_VARIABLE(make_iterator_range_fn, make_iterator_range)

    // TODO add specialization of range_cardinality for when we can determine the range is
    // infinite

    /// @}
} // namespace ranges

// The standard is inconsistent about whether these are classes or structs
RANGES_DIAGNOSTIC_PUSH
RANGES_DIAGNOSTIC_IGNORE_MISMATCHED_TAGS

/// \cond
namespace std
{
    template<typename I, typename S>
    struct tuple_size<::ranges::iterator_range<I, S>> : std::integral_constant<size_t, 2>
    {};

    template<typename I, typename S>
    struct tuple_element<0, ::ranges::iterator_range<I, S>>
    {
        using type = I;
    };

    template<typename I, typename S>
    struct tuple_element<1, ::ranges::iterator_range<I, S>>
    {
        using type = S;
    };

    template<typename I, typename S>
    struct tuple_size<::ranges::sized_iterator_range<I, S>>
      : std::integral_constant<size_t, 3>
    {};

    template<typename I, typename S>
    struct tuple_element<0, ::ranges::sized_iterator_range<I, S>>
    {
        using type = I;
    };

    template<typename I, typename S>
    struct tuple_element<1, ::ranges::sized_iterator_range<I, S>>
    {
        using type = S;
    };

    template<typename I, typename S>
    struct tuple_element<2, ::ranges::sized_iterator_range<I, S>>
    {
        using type = typename ::ranges::sized_iterator_range<I, S>::size_type;
    };
} // namespace std
/// \endcond

RANGES_DIAGNOSTIC_POP

#endif