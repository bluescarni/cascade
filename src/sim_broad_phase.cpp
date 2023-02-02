// Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <atomic>
#include <cassert>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <numeric>
#include <initializer_list>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <boost/safe_numerics/safe_integer.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <heyoka/detail/dfloat.hpp>

#include <cascade/detail/logging_impl.hpp>
#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"

#endif

#include "mdspan/mdspan"

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

namespace cascade
{

// Broad phase collision detection - i.e., collision
// detection between the AABBs of the particles' trajectories.
void sim::broad_phase_parallel()
{
    namespace stdex = std::experimental;

    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Cache a few quantities.
    const auto nparts = get_nparts();
    const auto nchunks = m_data->nchunks;
    const auto order = m_data->s_ta.get_order();

        if (!std::isfinite(m_conj_thresh * m_conj_thresh)) {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("A conjunction threshold of {} is too large and results in an overflow error", m_conj_thresh));
        // LCOV_EXCL_STOP
        }
    
    // Is conjunction detection activated globally?
    const auto with_conj = (m_conj_thresh != 0);

    // Reset the collision vector.
    m_data->coll_vec.clear();

    // Global counter for the total number of AABBs collisions
    // across all chunks.
    std::atomic<decltype(m_data->bp_coll[0].size())> tot_n_bp(0);

    // Views for accessing the sorted lbs/ubs data.
    using b_size_t = decltype(m_data->lbs.size());
    stdex::mdspan srt_lbs(std::as_const(m_data->srt_lbs).data(),
                          stdex::extents<b_size_t, stdex::dynamic_extent, stdex::dynamic_extent, 4u>(nchunks, nparts));
    stdex::mdspan srt_ubs(std::as_const(m_data->srt_ubs).data(),
                          stdex::extents<b_size_t, stdex::dynamic_extent, stdex::dynamic_extent, 4u>(nchunks, nparts));

    // View for accessing the indices vector.
    using idx_size_t = decltype(m_data->vidx.size());
    stdex::mdspan vidx(std::as_const(m_data->vidx).data(),
                       stdex::extents<idx_size_t, stdex::dynamic_extent, stdex::dynamic_extent>(nchunks, nparts));

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            // Fetch a reference to the tree.
            const auto &tree = m_data->bvh_trees[chunk_idx];

            // Fetch a reference to the AABB collision vector for the
            // current chunk and clear it out.
            auto &bp_cv = m_data->bp_coll[chunk_idx];
            bp_cv.clear();

            // Fetch a reference to the bp data cache for the current chunk.
            auto &bp_data_cache_ptr = m_data->bp_data_caches[chunk_idx];
            // NOTE: the pointer will require initialisation the first time
            // it is used.
            if (!bp_data_cache_ptr) {
                bp_data_cache_ptr
                    = std::make_unique<typename decltype(m_data->bp_data_caches)::value_type::element_type>();
            }
            auto &bp_data_cache = *bp_data_cache_ptr;

            // Fetch a reference to the chunk-specific narrow-phase caches.
            auto &np_cache_ptr = m_data->np_caches[chunk_idx];
            // NOTE: the pointer will require initialisation the first time
            // it is used.
            if (!np_cache_ptr) {
                np_cache_ptr = std::make_unique<typename decltype(m_data->np_caches)::value_type::element_type>();
            }
            auto &np_cache = *np_cache_ptr;

            // Fetch a reference to the detected conjunctions vector
            // for the current chunk and clear it out.
            auto &cl_conj_vec = m_data->conj_vecs[chunk_idx];
            cl_conj_vec.clear();

            // The time coordinate, relative to init_time, of
            // the chunk's begin/end.
            const auto [c_begin, c_end] = m_data->get_chunk_begin_end(chunk_idx, m_ct);
            const auto chunk_begin = heyoka::detail::dfloat<double>(c_begin);
            const auto chunk_end = heyoka::detail::dfloat<double>(c_end);

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &r2) {
                // Fetch local data for broad phase collision detection.
                std::unique_ptr<sim_data::bp_data> local_bp_data;
                if (bp_data_cache.try_pop(local_bp_data)) {
                    assert(local_bp_data);
                } else {
                    SPDLOG_LOGGER_DEBUG(logger, "Creating new local BP data");

                    local_bp_data = std::make_unique<sim_data::bp_data>();
                }

                // Fetch and setup local data for narrow-phase detection.
                std::unique_ptr<sim_data::np_data> pcaches;

                if (np_cache.try_pop(pcaches)) {
#if !defined(NDEBUG)
                    assert(pcaches);

                    for (auto &v : pcaches->pbuffers) {
                        assert(v.size() == order + 1u);
                    }

                    using safe_size_t = boost::safe_numerics::safe<decltype(pcaches->diff_input.size())>;
                    assert(pcaches->diff_input.size() == (order + 1u) * safe_size_t(6));
#endif
                } else {
                    SPDLOG_LOGGER_DEBUG(logger, "Creating new local polynomials for narrow phase collision detection");

                    // Init pcaches.
                    pcaches = std::make_unique<sim_data::np_data>();

                    for (auto &v : pcaches->pbuffers) {
                        v.resize(boost::numeric_cast<decltype(v.size())>(order + 1u));
                    }

                    using safe_size_t = boost::safe_numerics::safe<decltype(pcaches->diff_input.size())>;
                    pcaches->diff_input.resize((order + 1u) * safe_size_t(6));
                }

                // Prepare the local conjunction vector.
                auto &local_conj_vec = pcaches->local_conj_vec;
                local_conj_vec.clear();

                // Cache and clear the local list of collisions.
                auto &local_bp = local_bp_data->bp;
                local_bp.clear();

                // Cache the stack.
                // NOTE: this will be cleared at the beginning
                // of the traversal for each particle.
                auto &stack = local_bp_data->stack;

                // NOTE: the particle indices in this for loop refer to the
                // Morton-ordered data.
                for (auto pidx = r2.begin(); pidx != r2.end(); ++pidx) {
                    // Load the original particle index corresponding to
                    // particle pidx.
                    const auto orig_pidx = vidx(chunk_idx, pidx);

                    // Check if pidx is active for collisions and conjunctions.
                    const auto coll_active_pidx = m_data->coll_active[orig_pidx];
                    const auto conj_active_pidx = m_data->conj_active[orig_pidx];

                    // Reset the stack, and add the root node to it.
                    stack.clear();
                    stack.push_back(0);

                    // Cache the AABB of the current particle.
                    const auto x_lb = srt_lbs(chunk_idx, pidx, 0);
                    const auto y_lb = srt_lbs(chunk_idx, pidx, 1);
                    const auto z_lb = srt_lbs(chunk_idx, pidx, 2);
                    const auto r_lb = srt_lbs(chunk_idx, pidx, 3);

                    const auto x_ub = srt_ubs(chunk_idx, pidx, 0);
                    const auto y_ub = srt_ubs(chunk_idx, pidx, 1);
                    const auto z_ub = srt_ubs(chunk_idx, pidx, 2);
                    const auto r_ub = srt_ubs(chunk_idx, pidx, 3);

                    do {
                        // Pop a node.
                        const auto cur_node_idx = stack.back();
                        stack.pop_back();

                        // Fetch the AABB of the node.
                        const auto &cur_node = tree[static_cast<std::uint32_t>(cur_node_idx)];
                        const auto &n_lb = cur_node.lb;
                        const auto &n_ub = cur_node.ub;

                        // Check for overlap with the AABB of the current particle.
                        const bool overlap
                            = (x_ub >= n_lb[0] && x_lb <= n_ub[0]) && (y_ub >= n_lb[1] && y_lb <= n_ub[1])
                              && (z_ub >= n_lb[2] && z_lb <= n_ub[2]) && (r_ub >= n_lb[3] && r_lb <= n_ub[3]);

                        if (overlap) {
                            if (cur_node.left == -1) {
                                // Leaf node: mark pidx as a collision/conjunction
                                // candidate with all particles in the node, unless either:
                                // - pidx is colliding with itself (pidx == i), or
                                // - pidx > i, in order to avoid counting twice
                                //   the collisions (pidx, i) and (i, pidx), or
                                // - pidx and i are both inactive.
                                // NOTE: in case of a multi-particle leaf,
                                // the node's AABB is the composition of the AABBs
                                // of all particles in the node, and thus, in general,
                                // it is not strictly true that pidx will overlap with
                                // *all* particles in the node. In other words, we will
                                // be detecting AABB overlaps which are not actually there.
                                // This is ok, as they will be filtered out in the
                                // next stages of collision detection.
                                // NOTE: like in the outer loop, the index i here refers
                                // to the Morton-ordered data.
                                for (auto i = cur_node.begin; i != cur_node.end; ++i) {
                                    // Fetch index i in the original order.
                                    const auto orig_i = vidx(chunk_idx, i);

                                    if (orig_pidx >= orig_i) {
                                        continue;
                                    }

                                    // Check if i is active for collisions and conjunctions.
                                    const auto coll_active_i = m_data->coll_active[orig_i];
                                    const auto conj_active_i = m_data->conj_active[orig_i];

                                    if (coll_active_pidx || conj_active_pidx || coll_active_i || conj_active_i) {
                                        // TODO remove.
                                        local_bp.emplace_back(orig_pidx, orig_i);
                                        narrow_phase_pair(pcaches.get(),orig_pidx, orig_i,chunk_begin, chunk_end, logger );
                                    }
                                }
                            } else {
                                // Internal node: add both children to the
                                // stack and iterate.
                                stack.push_back(cur_node.left);
                                stack.push_back(cur_node.right);
                            }
                        }
                    } while (!stack.empty());
                }

                                // Atomically merge the local conjunction vector into the chunk-specific one.
                if (with_conj) {
                    cl_conj_vec.grow_by(local_conj_vec.begin(), local_conj_vec.end());
                }

                // Put the polynomials back into the caches.
                np_cache.push(std::move(pcaches));

                // Atomically merge the local bp into the chunk-local one.
                bp_cv.grow_by(local_bp.begin(), local_bp.end());

                // Put the local data back into the cache.
                bp_data_cache.push(std::move(local_bp_data));
            });

            // Update tot_n_bp with the data from the current chunk.
            tot_n_bp.fetch_add(bp_cv.size(), std::memory_order::relaxed);
        }
    });

        // NOTE: this is used only for logging purposes.
    std::optional<decltype(m_det_conj->size())> n_det_conjs;

    if (with_conj) {
        // If conjunction detection is active, we want to prepare
        // the global conjunction vector for the new detected conjunctions
        // which are currently stored in m_data->conj_vecs. The objective
        // is to avoid reallocating in the step() function, where
        // we need the noexcept guarantee.

        // We begin by determining how many conjunctions were detected.
        // NOTE: perhaps determining n_new_conj can be done in parallel
        // earlier. We just need to take care of avoiding overflow somehow.
        using safe_size_t = boost::safe_numerics::safe<decltype(m_det_conj->size())>;
        const auto n_new_conj = std::accumulate(m_data->conj_vecs.begin(), m_data->conj_vecs.end(), safe_size_t(0),
                                                [](const auto &acc, const auto &cur) { return acc + cur.size(); });

        // Do we have enough storage in m_det_conj to store the new conjunctions?
        if (m_det_conj->size() + n_new_conj > m_det_conj->capacity()) {
            // m_det_conj cannot store the new conjunctions without reallocating.
            // We thus prepare a new vector with twice the needed capacity.
            std::vector<conjunction> new_det_conj;
            new_det_conj.reserve(2u * (m_det_conj->size() + n_new_conj));

            // Copy over the existing conjunctions.
            new_det_conj.insert(new_det_conj.end(), m_det_conj->begin(), m_det_conj->end());

            // Assign the new conjunction vector.
            m_det_conj = std::make_shared<std::vector<conjunction>>(std::move(new_det_conj));
        }

        // Set the logging variable.
        n_det_conjs.emplace(n_new_conj);
    }

    logger->trace("Broad phase collision detection time: {}s", sw);

    logger->trace("Average number of AABB collisions per particle per chunk: {}",
                  static_cast<double>(tot_n_bp.load(std::memory_order::relaxed)) / static_cast<double>(nchunks)
                      / static_cast<double>(nparts));

    if (n_det_conjs) {
        logger->trace("Total number of conjunctions detected: {}", *n_det_conjs);
    }

#if !defined(NDEBUG)
    verify_broad_phase_parallel();
#endif
}

// Debug checks on the broad phase collision detection.
void sim::verify_broad_phase_parallel() const
{
    namespace stdex = std::experimental;

    const auto nparts = get_nparts();
    const auto nchunks = m_data->nchunks;

    // Don't run the check if there's too many particles.
    if (nparts > 10000u) {
        return;
    }

    // Views for accessing the lbs/ubs data.
    using b_size_t = decltype(m_data->lbs.size());
    stdex::mdspan lbs(std::as_const(m_data->lbs).data(),
                      stdex::extents<b_size_t, stdex::dynamic_extent, stdex::dynamic_extent, 4u>(nchunks, nparts));
    stdex::mdspan ubs(std::as_const(m_data->ubs).data(),
                      stdex::extents<b_size_t, stdex::dynamic_extent, stdex::dynamic_extent, 4u>(nchunks, nparts));

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            // Build a set version of the collision list
            // for fast lookup.
            std::set<std::pair<size_type, size_type>> coll_tree;
            for (const auto &p : m_data->bp_coll[chunk_idx]) {
                // Check that, for all collisions (i, j), i is always < j.
                assert(p.first < p.second);
                // Check that the collision pairs are unique.
                assert(coll_tree.emplace(p).second);
            }

            // A counter for the N**2 collision detection algorithm below.
            std::atomic<decltype(coll_tree.size())> coll_counter(0);

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &ri) {
                for (auto i = ri.begin(); i != ri.end(); ++i) {
                    const auto xi_lb = lbs(chunk_idx, i, 0);
                    const auto yi_lb = lbs(chunk_idx, i, 1);
                    const auto zi_lb = lbs(chunk_idx, i, 2);
                    const auto ri_lb = lbs(chunk_idx, i, 3);

                    const auto xi_ub = ubs(chunk_idx, i, 0);
                    const auto yi_ub = ubs(chunk_idx, i, 1);
                    const auto zi_ub = ubs(chunk_idx, i, 2);
                    const auto ri_ub = ubs(chunk_idx, i, 3);

                    // Check if i is active for collisions and conjunctions.
                    const auto coll_active_i = m_data->coll_active[i];
                    const auto conj_active_i = m_data->conj_active[i];

                    oneapi::tbb::parallel_for(
                        oneapi::tbb::blocked_range<size_type>(i + 1u, nparts), [&](const auto &rj) {
                            decltype(coll_tree.size()) loc_ncoll = 0;

                            for (auto j = rj.begin(); j != rj.end(); ++j) {
                                const auto xj_lb = lbs(chunk_idx, j, 0);
                                const auto yj_lb = lbs(chunk_idx, j, 1);
                                const auto zj_lb = lbs(chunk_idx, j, 2);
                                const auto rj_lb = lbs(chunk_idx, j, 3);

                                const auto xj_ub = ubs(chunk_idx, j, 0);
                                const auto yj_ub = ubs(chunk_idx, j, 1);
                                const auto zj_ub = ubs(chunk_idx, j, 2);
                                const auto rj_ub = ubs(chunk_idx, j, 3);

                                // Check if j is active for collisions and conjunctions.
                                const auto coll_active_j = m_data->coll_active[j];
                                const auto conj_active_j = m_data->conj_active[j];

                                const bool overlap
                                    = (xi_ub >= xj_lb && xi_lb <= xj_ub) && (yi_ub >= yj_lb && yi_lb <= yj_ub)
                                      && (zi_ub >= zj_lb && zi_lb <= zj_ub) && (ri_ub >= rj_lb && ri_lb <= rj_ub)
                                      && (coll_active_i || conj_active_i || coll_active_j || conj_active_j);

                                if (overlap) {
                                    // Overlap detected in the simple algorithm:
                                    // the collision must be present also
                                    // in the tree code.
                                    assert(coll_tree.find({i, j}) != coll_tree.end());
                                } else {
                                    // NOTE: the contrary is not necessarily
                                    // true: for multi-particle leaves, we
                                    // may detect overlaps that do not actually exist.
                                }

                                loc_ncoll += overlap;
                            }

                            coll_counter.fetch_add(loc_ncoll, std::memory_order::relaxed);
                        });
                }
            });

            // NOTE: in case of multi-particle leaves, we will have detected
            // non-existing AABBs overlaps. Thus, just require that the number
            // of collisions detected via the tree is at least as large
            // as the number of "true" collisions detected with the N**2 algorithm.
            assert(coll_tree.size() >= coll_counter.load());
        }
    });
}

} // namespace cascade
