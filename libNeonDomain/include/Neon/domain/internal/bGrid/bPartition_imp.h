#pragma once

#include "Neon/domain/internal/bGrid/bGrid.h"
#include "Neon/domain/internal/bGrid/bPartitionIndexSpace.h"

namespace Neon::domain::internal::bGrid {

template <typename T, int C>
bPartition<T, C>::bPartition()
    : mDataView(Neon::DataView::STANDARD),
      mLevel(0),
      mMem(nullptr),
      mMemParent(nullptr),
      mMemChild(nullptr),
      mCardinality(0),
      mNeighbourBlocks(nullptr),
      mOrigin(nullptr),
      mParentBlockID(nullptr),
      mParentLocalID(nullptr),
      mMask(nullptr),
      mMaskLowerLevel(nullptr),
      mChildBlockID(nullptr),
      mOutsideValue(0),
      mStencilNghIndex(nullptr),
      mRefFactors(nullptr),
      mSpacing(nullptr),
      mIsInSharedMem(false),
      mMemSharedMem(nullptr),
      mSharedNeighbourBlocks(nullptr),
      mStencilRadius(0)
{
}

template <typename T, int C>
bPartition<T, C>::bPartition(Neon::DataView  dataView,
                             int             level,
                             T*              mem,
                             T*              memParent,
                             T*              memChild,
                             int             cardinality,
                             uint32_t*       neighbourBlocks,
                             Neon::int32_3d* origin,
                             uint32_t*       parentBlockID,
                             Cell::Location* parentLocalID,
                             uint32_t*       mask,
                             uint32_t*       maskLowerLevel,
                             uint32_t*       childBlockID,
                             T               outsideValue,
                             nghIdx_t*       stencilNghIndex,
                             int*            refFactors,
                             int*            spacing)
    : mDataView(dataView),
      mLevel(level),
      mMem(mem),
      mMemParent(memParent),
      mMemChild(memChild),
      mCardinality(cardinality),
      mNeighbourBlocks(neighbourBlocks),
      mOrigin(origin),
      mParentBlockID(parentBlockID),
      mParentLocalID(parentLocalID),
      mMask(mask),
      mMaskLowerLevel(maskLowerLevel),
      mChildBlockID(childBlockID),
      mOutsideValue(outsideValue),
      mStencilNghIndex(stencilNghIndex),
      mRefFactors(refFactors),
      mSpacing(spacing),
      mIsInSharedMem(false),
      mMemSharedMem(nullptr),
      mSharedNeighbourBlocks(nullptr),
      mStencilRadius(0)
{
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::mapToGlobal(const Cell& cell) const -> Neon::index_3d
{
    Neon::index_3d ret = mOrigin[cell.mBlockID];
#ifdef NEON_PLACE_CUDA_DEVICE
    if constexpr (Cell::sUseSwirlIndex) {
        Cell::Location swirl = cell.toSwirl();
        ret.x += swirl.x;
        ret.y += swirl.y;
        ret.z += swirl.z;
    } else {
#endif
        const int sp = (mLevel == 0) ? 1 : mSpacing[mLevel - 1];
        ret.x += cell.mLocation.x * sp;
        ret.y += cell.mLocation.y * sp;
        ret.z += cell.mLocation.z * sp;
#ifdef NEON_PLACE_CUDA_DEVICE
    }
#endif
    return ret;
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::cardinality() const -> int
{
    return mCardinality;
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::operator()(const bCell& cell,
                                                               int          card) -> T&
{

    if (mIsInSharedMem) {
        return mMemSharedMem[shmemPitch(cell, card)];
    } else {
        return mMem[pitch(cell, card)];
    }
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::operator()(const bCell& cell,
                                                               int          card) const -> const T&
{
    if (!cell.mIsActive) {
        return mOutsideValue;
    }
    if (mIsInSharedMem) {
        return mMemSharedMem[shmemPitch(cell, card)];
    } else {
        return mMem[pitch(cell, card)];
    }
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::pitch(const Cell& cell, int card) const -> uint32_t
{
    //assumes SoA within the block i.e., AoSoA
    return
        //stride across all block before cell's block
        cell.mBlockID * cell.mBlockSize * cell.mBlockSize * cell.mBlockSize * mCardinality +
        //stride within the block
        cell.pitch(card);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::getRefFactor(const int level) const -> int
{
    return mRefFactors[level];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::getSpacing(const int level) const -> int
{
    return mSpacing[level];
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::childID(const Cell& cell) const -> uint32_t
{
    //return the child block id corresponding to this cell
    //the child block id lives at level mLevel-1

    const uint32_t childPitch =
        //stride across all block before cell's block
        cell.mBlockID *
            cell.mBlockSize * cell.mBlockSize * cell.mBlockSize +
        //stride within the block
        cell.mLocation.x +
        cell.mLocation.y * cell.mBlockSize +
        cell.mLocation.z * cell.mBlockSize * cell.mBlockSize;

    return mChildBlockID[childPitch];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::hasParent(const Cell& cell) const -> bool
{
    if (mMemParent) {
        return true;
    }
    return false;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::getChild(const Cell&   cell,
                                                             Neon::int8_3d child) const -> Cell
{
    Cell childCell;
    childCell.mBlockID = childID(cell);
    childCell.mBlockSize = mRefFactors[mLevel - 1];
    childCell.mLocation.x = child.x;
    childCell.mLocation.y = child.y;
    childCell.mLocation.z = child.z;
    childCell.mIsActive = childCell.computeIsActive(mMaskLowerLevel);
    return childCell;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::childVal(const Cell& childCell,
                                                             int         card) -> T&
{
    return mMemChild[pitch(childCell, card)];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::childVal(const Cell& childCell,
                                                             int         card) const -> const T&
{
    return mMemChild[pitch(childCell, card)];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::childVal(const Cell&   parent_cell,
                                                             Neon::int8_3d child,
                                                             int           card,
                                                             const T&      alternativeVal) const -> NghInfo<T>
{
    NghInfo<T> ret;
    ret.value = alternativeVal;
    ret.isValid = false;
    if (!parent_cell.mIsActive || !hasChildren(parent_cell)) {
        return ret;
    }

    Cell child_cell = getChild(parent_cell, child);

    if (!child_cell.mIsActive) {
        return ret;
    }

    ret.isValid = true;
    ret.value = childVal(child_cell, card);

    return ret;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::hasChildren(const Cell& cell) const -> bool
{
    if (mMemChild == nullptr || mMaskLowerLevel == nullptr || mLevel == 0) {
        return false;
    }
    if (childID(cell) == std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    return true;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::parent(const Cell& eId,
                                                           int         card) -> T&
{
    if (mMemParent != nullptr) {
        Cell parentCell;
        parentCell.mBlockID = mParentBlockID[eId.mBlockID];
        parentCell.mLocation = mParentLocalID[eId.mBlockID];
        parentCell.mBlockSize = mRefFactors[mLevel + 1];
        return mMemParent[pitch(parentCell, card)];
    }
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::setNghCell(const Cell&     cell,
                                                               const nghIdx_t& offset) const -> Cell
{
    Cell ngh_cell(cell.mLocation.x + offset.x,
                  cell.mLocation.y + offset.y,
                  cell.mLocation.z + offset.z);
    ngh_cell.mBlockSize = cell.mBlockSize;

    if (ngh_cell.mLocation.x < 0 || ngh_cell.mLocation.y < 0 || ngh_cell.mLocation.z < 0 ||
        ngh_cell.mLocation.x >= cell.mBlockSize || ngh_cell.mLocation.y >= cell.mBlockSize || ngh_cell.mLocation.z >= cell.mBlockSize) {

        //The neighbor is not in this block

        //Calculate the neighbor block ID and the local index within the neighbor block
        int16_3d block_offset(0, 0, 0);

        if (ngh_cell.mLocation.x < 0) {
            block_offset.x = -1;
            ngh_cell.mLocation.x += cell.mBlockSize;
        } else if (ngh_cell.mLocation.x >= cell.mBlockSize) {
            block_offset.x = 1;
            ngh_cell.mLocation.x -= cell.mBlockSize;
        }

        if (ngh_cell.mLocation.y < 0) {
            block_offset.y = -1;
            ngh_cell.mLocation.y += cell.mBlockSize;
        } else if (ngh_cell.mLocation.y >= cell.mBlockSize) {
            block_offset.y = 1;
            ngh_cell.mLocation.y -= cell.mBlockSize;
        }

        if (ngh_cell.mLocation.z < 0) {
            block_offset.z = -1;
            ngh_cell.mLocation.z += cell.mBlockSize;
        } else if (ngh_cell.mLocation.z >= cell.mBlockSize) {
            block_offset.z = 1;
            ngh_cell.mLocation.z -= cell.mBlockSize;
        }

        if (mSharedNeighbourBlocks != nullptr) {
            ngh_cell.mBlockID = mSharedNeighbourBlocks[Cell::getNeighbourBlockID(block_offset)];
        } else {
            ngh_cell.mBlockID = mNeighbourBlocks[26 * cell.mBlockID + Cell::getNeighbourBlockID(block_offset)];
        }

    } else {
        ngh_cell.mBlockID = cell.mBlockID;
    }
    return ngh_cell;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::nghVal(const Cell& eId,
                                                           uint8_t     nghID,
                                                           int         card,
                                                           const T&    alternativeVal) const -> NghInfo<T>
{
    nghIdx_t nghOffset = mStencilNghIndex[nghID];
    return nghVal(eId, nghOffset, card, alternativeVal);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::nghVal(const Cell&     cell,
                                                           const nghIdx_t& offset,
                                                           const int       card,
                                                           const T         alternativeVal) const -> NghInfo<T>
{
    NghInfo<T> ret;
    ret.value = alternativeVal;
    ret.isValid = false;
    if (!cell.mIsActive) {
        return ret;
    }


    if constexpr (Cell::sUseSwirlIndex) {
        Cell swirl_cell = cell.toSwirl();
        swirl_cell.mBlockSize = cell.mBlockSize;

        Cell ngh_cell = setNghCell(swirl_cell, offset);
        ngh_cell.mBlockSize = cell.mBlockSize;
        if (ngh_cell.mBlockID != std::numeric_limits<uint32_t>::max()) {
            //TODO maybe ngh_cell should be mapped to its memory layout
            ret.isValid = ngh_cell.computeIsActive(mMask);
            if (ret.isValid) {
                if (mIsInSharedMem) {
                    ngh_cell.mLocation.x = cell.mLocation.x + offset.x;
                    ngh_cell.mLocation.y = cell.mLocation.y + offset.y;
                    ngh_cell.mLocation.z = cell.mLocation.z + offset.z;
                }
                ret.value = this->operator()(ngh_cell, card);
            }
        }

    } else {
        Cell ngh_cell = setNghCell(cell, offset);
        ngh_cell.mBlockSize = cell.mBlockSize;
        if (ngh_cell.mBlockID != std::numeric_limits<uint32_t>::max()) {
            ret.isValid = ngh_cell.computeIsActive(mMask);
            if (ret.isValid) {
                if (mIsInSharedMem) {
                    ngh_cell.mLocation.x = cell.mLocation.x + offset.x;
                    ngh_cell.mLocation.y = cell.mLocation.y + offset.y;
                    ngh_cell.mLocation.z = cell.mLocation.z + offset.z;
                }
                ret.value = this->operator()(ngh_cell, card);
            }
        }
    }

    return ret;
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::shmemPitch(
    Cell      cell,
    const int card) const -> Cell::Location::Integer
{
    if constexpr (Cell::sUseSwirlIndex) {
        if (cell.mLocation.x >= 0 && cell.mLocation.x < cell.mBlockSize &&
            cell.mLocation.y >= 0 && cell.mLocation.y < cell.mBlockSize &&
            cell.mLocation.z >= -1 && cell.mLocation.z <= cell.mBlockSize) {

            if (cell.mLocation.z == -1) {
                cell.mLocation.z = cell.mBlockSize;
            } else if (cell.mLocation.z == 1) {
                cell.mLocation.z = cell.mBlockSize + 1;
            }
        } else {
            cell.mLocation.z += (cell.mLocation.z / 2) + 10;
        }


        return (2 * mStencilRadius + cell.mBlockSize) * (2 * mStencilRadius + cell.mBlockSize) * (2 * mStencilRadius + cell.mBlockSize) * static_cast<Cell::Location::Integer>(card) +
               cell.mLocation.x +
               cell.mLocation.y * cell.mBlockSize +
               cell.mLocation.z * cell.mBlockSize * cell.mBlockSize;

    } else {
        return (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize)) * (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize)) * (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize)) * static_cast<Cell::Location::Integer>(card) +
               //offset to this cell's data
               (cell.mLocation.x + mStencilRadius) + (cell.mLocation.y + mStencilRadius) * (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize)) + (cell.mLocation.z + mStencilRadius) * (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize)) * (2 * mStencilRadius + Cell::Location::Integer(cell.mBlockSize));
    }
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::loadInSharedMemory(
    [[maybe_unused]] const Cell&                cell,
    [[maybe_unused]] const nghIdx_t::Integer    stencilRadius,
    [[maybe_unused]] Neon::sys::ShmemAllocator& shmemAlloc) const -> void
{
#ifdef NEON_PLACE_CUDA_DEVICE
    mStencilRadius = stencilRadius;

    mMemSharedMem = shmemAlloc.alloc<T>((cell.mBlockSize + 2 * mStencilRadius) *
                                        (cell.mBlockSize + 2 * mStencilRadius) *
                                        (cell.mBlockSize + 2 * mStencilRadius) * mCardinality);
    //load the block itself in shared memory
    Cell shmemcell(cell.mLocation.x, cell.mLocation.y, cell.mLocation.z);


    //load the 6 faces from neighbor blocks in shared memory
    auto load_ngh = [&](const nghIdx_t& offset, int card) {
        NghInfo<T> ngh = nghVal(cell, offset, card, mOutsideValue);
        shmemcell.mLocation.x = cell.mLocation.x + offset.x;
        shmemcell.mLocation.y = cell.mLocation.y + offset.y;
        shmemcell.mLocation.z = cell.mLocation.z + offset.z;
        mMemSharedMem[shmemPitch(shmemcell, card)] = ngh.value;
    };


    /*__shared__ uint32_t sNeighbour[26];
    mSharedNeighbourBlocks = sNeighbour;
    {
        Cell::Location::Integer tid = cell.getLocal1DID();
        for (int i = 0; i < 26; i += cell.mBlockSize * cell.mBlockSize * cell.mBlockSize) {
            mSharedNeighbourBlocks[i] = mNeighbourBlocks[26 * cell.mBlockID + i];
        }
    }*/

    nghIdx_t offset;
#pragma unroll 2
    for (int card = 0; card < mCardinality; ++card) {
        mMemSharedMem[shmemPitch(shmemcell, card)] = this->operator()(cell, card);

        for (nghIdx_t::Integer r = 1; r <= mStencilRadius; ++r) {
            //face (x, y, -z)
            if (cell.mLocation.z == 0) {
                offset.x = 0;
                offset.y = 0;
                offset.z = -r;
                load_ngh(offset, card);
            }

            //face (x, y, +z)
            if (cell.mLocation.z == cell.mBlockSize - 1) {
                offset.x = 0;
                offset.y = 0;
                offset.z = r;
                load_ngh(offset, card);
            }

            //face (x, -y, z)
            if (cell.mLocation.y == 0) {
                offset.x = 0;
                offset.y = -r;
                offset.z = 0;
                load_ngh(offset, card);
            }

            //face (x, +y, z)
            if (cell.mLocation.y == cell.mBlockSize - 1) {
                offset.x = 0;
                offset.y = r;
                offset.z = 0;
                load_ngh(offset, card);
            }

            //face (-x, y, z)
            if (cell.mLocation.x == 0) {
                offset.x = -r;
                offset.y = 0;
                offset.z = 0;
                load_ngh(offset, card);
            }

            //face (+x, y, z)
            if (cell.mLocation.x == cell.mBlockSize - 1) {
                offset.x = r;
                offset.y = 0;
                offset.z = 0;
                load_ngh(offset, card);
            }


            //load the 12 edges

            //edges along x-axis
            // edge (x, -y, -z)
            if (cell.mLocation.y == 0 && cell.mLocation.z == 0) {
                offset.x = 0;
                offset.y = -r;
                offset.z = -r;
                load_ngh(offset, card);
            }
            // edge (x, +y, -z)
            if (cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == 0) {
                offset.x = 0;
                offset.y = r;
                offset.z = -r;
                load_ngh(offset, card);
            }
            // edge (x, -y, +z)
            if (cell.mLocation.y == 0 && cell.mLocation.z == cell.mBlockSize - 1) {
                offset.x = 0;
                offset.y = -r;
                offset.z = r;
                load_ngh(offset, card);
            }
            // edge (x, +y, +z)
            if (cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == cell.mBlockSize - 1) {
                offset.x = 0;
                offset.y = r;
                offset.z = r;
                load_ngh(offset, card);
            }


            //edges along y-axis
            // edge (-x, y, -z)
            if (cell.mLocation.x == 0 && cell.mLocation.z == 0) {
                offset.x = -r;
                offset.y = 0;
                offset.z = -r;
                load_ngh(offset, card);
            }
            // edge (-x, y, +z)
            if (cell.mLocation.x == 0 && cell.mLocation.z == cell.mBlockSize - 1) {
                offset.x = -r;
                offset.y = 0;
                offset.z = r;
                load_ngh(offset, card);
            }
            // edge (+x, y, -z)
            if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.z == 0) {
                offset.x = r;
                offset.y = 0;
                offset.z = -r;
                load_ngh(offset, card);
            }
            // edge (+x, y, -z)
            if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.z == cell.mBlockSize - 1) {
                offset.x = r;
                offset.y = 0;
                offset.z = r;
                load_ngh(offset, card);
            }


            //edges along z-axis
            // edge (-x, -y, z)
            if (cell.mLocation.x == 0 && cell.mLocation.y == 0) {
                offset.x = -r;
                offset.y = -r;
                offset.z = 0;
                load_ngh(offset, card);
            }
            // edge (-x, +y, z)
            if (cell.mLocation.x == 0 && cell.mLocation.y == cell.mBlockSize - 1) {
                offset.x = -r;
                offset.y = r;
                offset.z = 0;
                load_ngh(offset, card);
            }
            // edge (+x, -y, z)
            if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == 0) {
                offset.x = r;
                offset.y = -r;
                offset.z = 0;
                load_ngh(offset, card);
            }
            // edge (+x, +y, z)
            if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == cell.mBlockSize - 1) {
                offset.x = r;
                offset.y = r;
                offset.z = 0;
                load_ngh(offset, card);
            }
        }

        //load the 8 corner

        //0,0,0
        for (Cell::Location::Integer z = -mStencilRadius; z <= -1; ++z) {
            for (Cell::Location::Integer y = -mStencilRadius; y <= -1; ++y) {
                for (Cell::Location::Integer x = -mStencilRadius; x <= -1; ++x) {

                    //0,0,0
                    if (cell.mLocation.x == 0 && cell.mLocation.y == 0 && cell.mLocation.z == 0) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,0,0
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == 0 && cell.mLocation.z == 0) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //0,1,0
                    if (cell.mLocation.x == 0 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == 0) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,1,0
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == 0) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }


                    //0,0,1
                    if (cell.mLocation.x == 0 && cell.mLocation.y == 0 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,0,1
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == 0 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //0,1,1
                    if (cell.mLocation.x == 0 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }

                    //1,1,1
                    if (cell.mLocation.x == cell.mBlockSize - 1 && cell.mLocation.y == cell.mBlockSize - 1 && cell.mLocation.z == cell.mBlockSize - 1) {
                        offset.x = -1 * static_cast<nghIdx_t::Integer>(x);
                        offset.y = -1 * static_cast<nghIdx_t::Integer>(y);
                        offset.z = -1 * static_cast<nghIdx_t::Integer>(z);
                        load_ngh(offset, card);
                    }
                }
            }
        }
    }


    __syncthreads();
    mIsInSharedMem = true;
#endif
};

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::loadInSharedMemoryAsync(
    [[maybe_unused]] const Cell&                cell,
    [[maybe_unused]] const nghIdx_t::Integer    stencilRadius,
    [[maybe_unused]] Neon::sys::ShmemAllocator& shmemAlloc) const -> void
{
#ifdef NEON_PLACE_CUDA_DEVICE
    //TODO only works on cardinality 1 for now
    assert(mCardinality == 1);
    //TODO only works on stencil 1 for now
    assert(stencilRadius == 1);

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    mStencilRadius = stencilRadius;

    __shared__ uint32_t sNeighbour[26];
    mSharedNeighbourBlocks = sNeighbour;

    Neon::sys::loadSharedMemAsync(
        block,
        mNeighbourBlocks + 26 * cell.mBlockID,
        26,
        mSharedNeighbourBlocks,
        true);


    mMemSharedMem = shmemAlloc.alloc<T>((cell.mBlockSize + 2 * mStencilRadius) *
                                        (cell.mBlockSize + 2 * mStencilRadius) *
                                        (cell.mBlockSize + 2 * mStencilRadius) * mCardinality);


    Cell::Location::Integer shmem_offset = 0;

    auto load = [&](const int16_3d block_offset,
                    const uint32_t src_offset,
                    const uint32_t size) {
        uint32_t ngh_block_id = mSharedNeighbourBlocks[Cell::getNeighbourBlockID(block_offset)];
        assert(ngh_block_id != cell.mBlockID);
        if (ngh_block_id != std::numeric_limits<uint32_t>::max()) {
            Neon::sys::loadSharedMemAsync(
                block,
                mMem + ngh_block_id * cell.mBlockSize * cell.mBlockSize * cell.mBlockSize * mCardinality + src_offset,
                //     ^^start of this block                                                                ^^ offset from the start
                size,
                mMemSharedMem + shmem_offset,
                false);
            shmem_offset += size;
        }
    };


    //load the interior
    Neon::sys::loadSharedMemAsync(
        block,
        mMem + cell.mBlockID * cell.mBlockSize * cell.mBlockSize * cell.mBlockSize * mCardinality,
        cell.mBlockSize * cell.mBlockSize * cell.mBlockSize,
        mMemSharedMem,
        false);
    shmem_offset += cell.mBlockSize * cell.mBlockSize * cell.mBlockSize;


    if (mStencilRadius > 0) {
        int16_3d block_offset(0, 0, 0);

        //load -Z faces
        block_offset.x = 0;
        block_offset.y = 0;
        block_offset.z = -1;
        load(block_offset,
             cell.mBlockSize * cell.mBlockSize * (cell.mBlockSize - 1),
             cell.mBlockSize * cell.mBlockSize);


        //load +Z faces
        block_offset.x = 0;
        block_offset.y = 0;
        block_offset.z = 1;
        load(block_offset,
             0,
             cell.mBlockSize * cell.mBlockSize);


        for (int z = 0; z < cell.mBlockSize; ++z) {
            // load strips from -Y
            block_offset.x = 0;
            block_offset.y = -1;
            block_offset.z = 0;
            load(block_offset,
                 z * cell.mBlockSize * cell.mBlockSize + 14,
                 cell.mBlockSize);

            // load strips from +Y
            block_offset.x = 0;
            block_offset.y = 1;
            block_offset.z = 0;
            load(block_offset,
                 z * cell.mBlockSize * cell.mBlockSize + 0,
                 cell.mBlockSize);

            // load strips from -X
            block_offset.x = -1;
            block_offset.y = 0;
            block_offset.z = 0;
            load(block_offset,
                 z * cell.mBlockSize * cell.mBlockSize + 7,
                 cell.mBlockSize);


            // load strips from +X
            block_offset.x = 1;
            block_offset.y = 0;
            block_offset.z = 0;
            load(block_offset,
                 z * cell.mBlockSize * cell.mBlockSize + 21,
                 cell.mBlockSize - 1);
            load(block_offset,
                 z * cell.mBlockSize * cell.mBlockSize + 0,
                 1);
        }
    }

    //wait for all memory to arrive
    cg::wait(block);
    //do we really need a sync here or wait() is enough
    block.sync();
    mIsInSharedMem = true;
#endif
};

}  // namespace Neon::domain::internal::bGrid