#pragma once

#include "./Methods.h"
#include "CellType.h"
#include "D3Q19.h"
#include "DeviceD3Q19.h"
#include "DeviceD3QXX.h"
#include "Methods.h"
#include "Neon/Neon.h"
#include "Neon/set/Containter.h"

/**
 * Specialization for D3Q19
 */
template <typename Precision_, typename Grid_, typename Lattice_, Collision CollisionId>
struct ContainerFactoryD3QXX
{
    using Lattice = Lattice_;
    using Precision = Precision_;
    using Compute = typename Precision::Compute;
    using Storage = typename Precision::Storage;
    using Grid = Grid_;

    using PopField = typename Grid::template Field<Storage, Lattice::Q>;
    using CellTypeField = typename Grid::template Field<CellType, 1>;

    using Idx = typename PopField::Idx;
    using Rho = typename Grid::template Field<Storage, 1>;
    using U = typename Grid::template Field<Storage, 3>;

    //    using PullFunctions = pull::DeviceD3Q19<Precision, Grid>;
    //    using CommonFunctions = common::DeviceD3Q19<Precision, Grid>;
    using Device = DeviceD3QXX<Precision, Grid, Lattice>;

    struct AA
    {
        struct Even
        {

            static auto
            iteration(const CellTypeField& cellTypeField /*! Cell type field     */,
                      const Compute        omega /*!         LBM omega parameter */,
                      NEON_IO PopField&    fpopField /*!     Output Population field */)
                -> Neon::set::Container
            {
                Neon::set::Container container = fpopField.getGrid().newContainer(
                    "D3Q19_TwoPop_Pull",
                    [&, omega](Neon::set::Loader& L) -> auto {
                        auto&                          fpop = L.load(fpopField);
                        const auto&                    cellInfoPartition = L.load(cellTypeField);
                        [[maybe_unused]] const Compute beta = omega * 0.5;
                        [[maybe_unused]] const Compute invBeta = 1.0 / beta;

                        return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                            CellType cellInfo = cellInfoPartition(gidx, 0);
                            if (cellInfo.classification == CellType::bulk) {

                                Storage popIn[Lattice::Q];
                                Device::Common::localLoad(gidx, fpop, NEON_OUT popIn);

                                Compute                rho;
                                std::array<Compute, 3> u{.0, .0, .0};
                                Device::Common::macroscopic(popIn, NEON_OUT rho, NEON_OUT u);

                                Compute usqr = 1.5 * (u[0] * u[0] +
                                                      u[1] * u[1] +
                                                      u[2] * u[2]);

                                if constexpr (CollisionId == Collision::bgk) {
                                    Device::Common::collideBgkUnrolled(rho, u,
                                                                       usqr, omega,
                                                                       NEON_IO popIn);
                                }
                                if constexpr (CollisionId == Collision::kbc) {
                                    Device::Common::collideKBCUnrolled(rho, u,
                                                                       usqr, omega,
                                                                       invBeta,
                                                                       NEON_IO popIn);
                                }
                                Device::Common::localStore(gidx, popIn, fpop);
                            }
                        };
                    });
                return container;
            }

            static auto
            computeRhoAndU([[maybe_unused]] const PopField& fInField /*!   inpout population field */,
                           const CellTypeField&             cellTypeField /*!       Cell type field     */,
                           Rho&                             rhoField /*!  output Population field */,
                           U&                               uField /*!  output Population field */)

                -> Neon::set::Container
            {
                return Pull::computeRhoAndU(fInField, cellTypeField, rhoField, uField);
            }
        };
        struct Odd
        {
            static auto
            iteration(Neon::set::StencilSemantic stencilSemantic,
                      const CellTypeField&       cellTypeField /*! Cell type field     */,
                      const Compute              omega /*!         LBM omega parameter */,
                      NEON_IO PopField&          fPopField /*!     Output Population field */)
                -> Neon::set::Container
            {
                Neon::set::Container container = fPopField.getGrid().newContainer(
                    "LBM-iteration",
                    [=](Neon::set::Loader& L) -> auto {
                        auto&       fPop = L.load(fPopField,
                                                  Neon::Pattern::STENCIL, stencilSemantic);
                        const auto& cellInfoPartition = L.load(cellTypeField);

                        [[maybe_unused]] const Compute beta = omega * 0.5;
                        [[maybe_unused]] const Compute invBeta = 1.0 / beta;

                        return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                            CellType cellInfo = cellInfoPartition(gidx, 0);
                            if (cellInfo.classification == CellType::bulk) {

                                Storage popIn[Lattice::Q];
                                Device::Pull::pullStream(gidx, cellInfo.wallNghBitflag, fPop, NEON_OUT popIn);

                                Compute                rho;
                                std::array<Compute, 3> u{.0, .0, .0};
                                Device::Common::macroscopic(popIn,
                                                            NEON_OUT rho, NEON_OUT u);

                                Compute usqr = 1.5 * (u[0] * u[0] +
                                                      u[1] * u[1] +
                                                      u[2] * u[2]);


                                if constexpr (CollisionId == Collision::bgk) {
                                    Device::Common::collideBgkUnrolled(rho, u,
                                                                       usqr, omega,
                                                                       NEON_IO popIn);
                                }
                                if constexpr (CollisionId == Collision::kbc) {
                                    Device::Common::collideKBCUnrolled(rho, u,
                                                                       usqr, omega,
                                                                       invBeta,
                                                                       NEON_IO popIn);
                                }
                                Device::Push::pushStream(gidx, cellInfo.wallNghBitflag, popIn, NEON_OUT fPop);
                            }
                        };
                    });
                return container;
            }

            static auto
            computeRhoAndU([[maybe_unused]] const PopField& fInField /*!   inpout population field */,
                           const CellTypeField&             cellTypeField /*!       Cell type field     */,
                           Rho&                             rhoField /*!  output Population field */,
                           U&                               uField /*!  output Population field */)

                -> Neon::set::Container
            {
                return Push::computeRhoAndU(fInField, cellTypeField, rhoField, uField);
            }
        };
    };

    struct Pull
    {
        static auto
        iteration(Neon::set::StencilSemantic stencilSemantic,
                  const PopField&            fInField /*!      Input population field */,
                  const CellTypeField&       cellTypeField /*! Cell type field     */,
                  const Compute              omega /*!         LBM omega parameter */,
                  PopField&                  fOutField /*!     Output Population field */)
            -> Neon::set::Container
        {
            Neon::set::Container container = fInField.getGrid().newContainer(
                "D3Q19_TwoPop_Pull",
                [&, omega](Neon::set::Loader& L) -> auto {
                    auto&                          fIn = L.load(fInField,
                                                                Neon::Pattern::STENCIL, stencilSemantic);
                    auto&                          fOut = L.load(fOutField);
                    const auto&                    cellInfoPartition = L.load(cellTypeField);
                    [[maybe_unused]] const Compute beta = omega * 0.5;
                    [[maybe_unused]] const Compute invBeta = 1.0 / beta;

                    return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                        CellType cellInfo = cellInfoPartition(gidx, 0);
                        if (cellInfo.classification == CellType::bulk) {

                            Storage popIn[Lattice::Q];
                            Device::Pull::pullStream(gidx, cellInfo.wallNghBitflag, fIn, NEON_OUT popIn);

                            Compute                rho;
                            std::array<Compute, 3> u{.0, .0, .0};
                            Device::Common::macroscopic(popIn, NEON_OUT rho, NEON_OUT u);

                            Compute usqr = 1.5 * (u[0] * u[0] +
                                                  u[1] * u[1] +
                                                  u[2] * u[2]);

                            if constexpr (CollisionId == Collision::bgk) {
                                Device::Common::collideBgkUnrolled(rho, u,
                                                                   usqr, omega,
                                                                   NEON_IO popIn);
                            }
                            if constexpr (CollisionId == Collision::kbc) {
                                Device::Common::collideKBCUnrolled(rho, u,
                                                                   usqr, omega,
                                                                   invBeta,
                                                                   NEON_IO popIn);
                            }
                            Device::Common::localStore(gidx, popIn, fOut);
                        }
                    };
                });
            return container;
        }

        static auto
        localCollide(const PopField&      fInField /*!      Input population field */,
                     const CellTypeField& cellTypeField /*! Cell type field     */,
                     const Compute        omega /*!         LBM omega parameter */,
                     PopField&            fOutField /*!     Output Population field */)
            -> Neon::set::Container
        {
            Neon::set::Container container = fInField.getGrid().newContainer(
                "D3Q19_TwoPop_Pull",
                [&, omega](Neon::set::Loader& L) -> auto {
                    auto&                          fIn = L.load(fInField);
                    auto&                          fOut = L.load(fOutField);
                    const auto&                    cellInfoPartition = L.load(cellTypeField);
                    [[maybe_unused]] const Compute beta = omega * 0.5;
                    [[maybe_unused]] const Compute invBeta = 1.0 / beta;

                    return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                        CellType cellInfo = cellInfoPartition(gidx, 0);
                        if (cellInfo.classification == CellType::bulk) {

                            Storage popIn[Lattice::Q];
                            Device::Common::localLoad(gidx, fIn, NEON_OUT popIn);

                            Compute                rho;
                            std::array<Compute, 3> u{.0, .0, .0};
                            Device::Common::macroscopic(popIn, NEON_OUT rho, NEON_OUT u);

                            Compute usqr = 1.5 * (u[0] * u[0] +
                                                  u[1] * u[1] +
                                                  u[2] * u[2]);

                            if constexpr (CollisionId == Collision::bgk) {
                                Device::Common::collideBgkUnrolled(rho, u,
                                                                   usqr, omega,
                                                                   NEON_IO popIn);
                            }
                            if constexpr (CollisionId == Collision::kbc) {
                                Device::Common::collideKBCUnrolled(rho, u,
                                                                   usqr, omega,
                                                                   invBeta,
                                                                   NEON_IO popIn);
                            }
                            Device::Common::localStore(gidx, popIn, fOut);
                        }
                    };
                });
            return container;
        }

        static auto
        computeRhoAndU([[maybe_unused]] const PopField& fInField /*!   inpout population field */,
                       const CellTypeField&             cellTypeField /*!       Cell type field     */,
                       Rho&                             rhoField /*!  output Population field */,
                       U&                               uField /*!  output Population field */)

            -> Neon::set::Container
        {

            Neon::set::Container container =
                fInField.getGrid().newContainer(
                    "LBM_iteration",
                    [&](Neon::set::Loader& L) -> auto {
                        auto& fIn = L.load(fInField,
                                           Neon::Pattern::STENCIL);
                        auto& rhoXpu = L.load(rhoField);
                        auto& uXpu = L.load(uField);

                        const auto& cellInfoPartition = L.load(cellTypeField);

                        return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                            CellType               cellInfo = cellInfoPartition(gidx, 0);
                            Compute                rho = 0;
                            std::array<Compute, 3> u{.0, .0, .0};

                            Storage popIn[Lattice::Q];

                            if (cellInfo.classification == CellType::bulk) {
                                Storage popIn[Lattice::Q];
                                Device::Pull::pullStream(gidx, cellInfo.wallNghBitflag, fIn, NEON_OUT popIn);
                                Device::Common::macroscopic(popIn, NEON_OUT rho, NEON_OUT u);
                            } else {
                                Device::Common::localLoad(gidx, fIn, NEON_OUT popIn);
                                if (cellInfo.classification == CellType::movingWall) {
                                    rho = 1.0;
                                    u = std::array<Compute, 3>{static_cast<Compute>(popIn[0]) / static_cast<Compute>(6. * 1. / 18.),
                                                               static_cast<Compute>(popIn[1]) / static_cast<Compute>(6. * 1. / 18.),
                                                               static_cast<Compute>(popIn[2]) / static_cast<Compute>(6. * 1. / 18.)};
                                }
                            }

                            rhoXpu(gidx, 0) = static_cast<Storage>(rho);
                            uXpu(gidx, 0) = static_cast<Storage>(u[0]);
                            uXpu(gidx, 1) = static_cast<Storage>(u[1]);
                            uXpu(gidx, 2) = static_cast<Storage>(u[2]);
                        };
                    });
            return container;
        }
    };
    struct Push
    {
        static auto
        iteration(Neon::set::StencilSemantic stencilSemantic,
                  const PopField&            fInField /*!      Input population field */,
                  const CellTypeField&       cellTypeField /*! Cell type field     */,
                  const Compute              omega /*!         LBM omega parameter */,
                  PopField&                  fOutField /*!     Output Population field */)
            -> Neon::set::Container
        {
            Neon::set::Container container = fInField.getGrid().newContainer(
                "LBM-iteration",
                [=](Neon::set::Loader& L) -> auto {
                    auto&       fIn = L.load(fInField,
                                             Neon::Pattern::STENCIL, stencilSemantic);
                    auto        fOut = L.load(fOutField);
                    const auto& cellInfoPartition = L.load(cellTypeField);

                    [[maybe_unused]] const Compute beta = omega * 0.5;
                    [[maybe_unused]] const Compute invBeta = 1.0 / beta;

                    return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                        CellType cellInfo = cellInfoPartition(gidx, 0);
                        if (cellInfo.classification == CellType::bulk) {

                            Storage popIn[Lattice::Q];
                            Device::Common::localLoad(gidx, fIn, NEON_OUT popIn);

                            Compute                rho;
                            std::array<Compute, 3> u{.0, .0, .0};
                            Device::Common::macroscopic(popIn,
                                                        NEON_OUT rho, NEON_OUT u);

                            Compute usqr = 1.5 * (u[0] * u[0] +
                                                  u[1] * u[1] +
                                                  u[2] * u[2]);


                            if constexpr (CollisionId == Collision::bgk) {
                                Device::Common::collideBgkUnrolled(rho, u,
                                                                   usqr, omega,
                                                                   NEON_IO popIn);
                            }
                            if constexpr (CollisionId == Collision::kbc) {
                                Device::Common::collideKBCUnrolled(rho, u,
                                                                   usqr, omega,
                                                                   invBeta,
                                                                   NEON_IO popIn);
                            }
                            Device::Push::pushStream(gidx, cellInfo.wallNghBitflag, popIn, NEON_OUT fOut);
                        }
                    };
                });
            return container;
        }

        static auto
        computeRhoAndU([[maybe_unused]] const PopField& fInField /*!   inpout population field */,
                       const CellTypeField&             cellTypeField /*!       Cell type field     */,
                       Rho&                             rhoField /*!  output Population field */,
                       U&                               uField /*!  output Population field */)

            -> Neon::set::Container
        {

            Neon::set::Container container =
                fInField.getGrid().newContainer(
                    "LBM_iteration",
                    [&](Neon::set::Loader& L) -> auto {
                        auto& fIn = L.load(fInField,
                                           Neon::Pattern::STENCIL);
                        auto& rhoXpu = L.load(rhoField);
                        auto& uXpu = L.load(uField);

                        const auto& cellInfoPartition = L.load(cellTypeField);

                        return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                            CellType               cellInfo = cellInfoPartition(gidx, 0);
                            Compute                rho = 0;
                            std::array<Compute, 3> u{.0, .0, .0};

                            Storage popIn[Lattice::Q];
                            Device::Common::localLoad(gidx, fIn, NEON_OUT popIn);

                            if (cellInfo.classification == CellType::bulk) {
                                Device::Common::macroscopic(popIn, NEON_OUT rho, NEON_OUT u);
                            } else {
                                if (cellInfo.classification == CellType::movingWall) {
                                    rho = 1.0;
                                    u = std::array<Compute, 3>{static_cast<Compute>(popIn[0]) / static_cast<Compute>(6. * 1. / 18.),
                                                               static_cast<Compute>(popIn[1]) / static_cast<Compute>(6. * 1. / 18.),
                                                               static_cast<Compute>(popIn[2]) / static_cast<Compute>(6. * 1. / 18.)};
                                }
                            }

                            rhoXpu(gidx, 0) = static_cast<Storage>(rho);
                            uXpu(gidx, 0) = static_cast<Storage>(u[0]);
                            uXpu(gidx, 1) = static_cast<Storage>(u[1]);
                            uXpu(gidx, 2) = static_cast<Storage>(u[2]);
                        };
                    });
            return container;
        }
    };
    struct Common
    {

        template <lbm::Method method_>
        static auto
        iteration([[maybe_unused]] Neon::set::StencilSemantic stencilSemantic,
                  [[maybe_unused]] const PopField             fInField /*!      Input population field */,
                  [[maybe_unused]] const CellTypeField&       cellTypeField /*! Cell type field     */,
                  [[maybe_unused]] const Compute              omega /*!         LBM omega parameter */,
                  [[maybe_unused]] PopField                   fOutField /*!     Output Population field */)
            -> Neon::set::Container
        {
            if constexpr (method_ == lbm::Method::push) {
                using FactoryPush = push::ContainerFactory<Precision_,
                                                           D3Q19<Precision_>,
                                                           Grid_>;
                return FactoryPush::iteration(stencilSemantic,
                                              fInField,
                                              cellTypeField,
                                              omega,
                                              fOutField);
            }
            if constexpr (method_ == lbm::Method::pull) {
                using FactoryPull = pull::ContainerFactory<Precision_,
                                                           D3Q19<Precision_>,
                                                           Grid_>;
                return FactoryPull::iteration(stencilSemantic,
                                              fInField,
                                              cellTypeField,
                                              omega,
                                              fOutField);
            }
            NEON_DEV_UNDER_CONSTRUCTION("");
        }


        static auto
        computeWallNghMask(const CellTypeField& infoInField,
                           CellTypeField&       infoOutpeField)

            -> Neon::set::Container
        {
            Neon::set::Container container = infoInField.getGrid().newContainer(
                "LBM_iteration",
                [&](Neon::set::Loader& L) -> auto {
                    auto& infoIn = L.load(infoInField, Neon::Pattern::STENCIL);
                    auto& infoOut = L.load(infoOutpeField);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                        CellType cellType = infoIn(gidx, 0);
                        cellType.wallNghBitflag = 0;

                        if (cellType.classification == CellType::bulk) {
                            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto fwdRegIdx) {
                                using M = typename Lattice::template RegisterMapper<fwdRegIdx>;
                                if constexpr (M::centerMemQ != M::fwdMemQ) {
                                    CellType nghCellType = infoIn.template getNghData<M::fwdMemQX, M::fwdMemQY, M::fwdMemQZ>(gidx, 0, CellType::undefined)();
                                    if (nghCellType.classification == CellType::bounceBack ||
                                        nghCellType.classification == CellType::movingWall) {
                                        cellType.wallNghBitflag = cellType.wallNghBitflag | ((uint32_t(1) << M::fwdMemQ));
                                    }
                                }
                            });
                            infoOut(gidx, 0) = cellType;
                        }
                    };
                });
            return container;
        }


        template <typename UserLambda>
        static auto
        userSettingBc(UserLambda     userLambda,
                      PopField&      pField,
                      CellTypeField& cellTypeField /*! Cell type field     */)
            -> Neon::set::Container
        {
            Neon::set::Container container = pField.getGrid().newContainer(
                "UserSettingBc",
                [&](Neon::set::Loader& L) -> auto {
                    auto& p = L.load(pField, Neon::Pattern::MAP);
                    auto& flag = L.load(cellTypeField, Neon::Pattern::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                        const auto               globalIdx = p.getGlobalIndex(gidx);
                        Storage                  pValues[Lattice::Q];
                        CellType::Classification cellClass;
                        userLambda(globalIdx, pValues, cellClass);

                        CellType flagVal(cellClass);
                        flag(gidx, 0) = flagVal;

                        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                            using M = typename Lattice::template RegisterMapper<q>;
                            p(gidx, M::fwdMemQ) = pValues[M::fwdRegQ];
                        });
                    };
                });
            return container;
        }

        static auto
        copyPopulation(PopField& fInField,
                       PopField& foutField)
            -> Neon::set::Container
        {
            Neon::set::Container container = fInField.getGrid().newContainer(
                "LBM_iteration",
                [&](Neon::set::Loader& L) -> auto {
                    auto const& pIn = L.load(fInField, Neon::Pattern::MAP);
                    auto&       pOut = L.load(foutField, Neon::Pattern::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                            pOut(gidx, q) = pIn(gidx, q);
                        });
                    };
                });
            return container;
        }


        static auto
        problemSetup(PopField&       fInField /*!   inpout population field */,
                     PopField&       fOutField,
                     CellTypeField&  cellTypeField,
                     Neon::double_3d ulid,
                     double          ulb)

            -> Neon::set::Container
        {
            Neon::set::Container container = fInField.getGrid().newContainer(
                "LBM_iteration",
                [&, ulid, ulb](Neon::set::Loader& L) -> auto {
                    auto& fIn = L.load(fInField, Neon::Pattern::MAP);
                    auto& fOut = L.load(fOutField, Neon::Pattern::MAP);
                    auto& cellInfoPartition = L.load(cellTypeField, Neon::Pattern::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                        const auto globalIdx = fIn.getGlobalIndex(gidx);
                        const auto domainDim = fIn.getDomainSize();

                        CellType flagVal;
                        flagVal.classification = CellType::bulk;
                        flagVal.wallNghBitflag = 0;

                        typename Lattice::Precision::Storage popVal = 0;

                        if (globalIdx.x == 0 || globalIdx.x == domainDim.x - 1 ||
                            globalIdx.y == 0 || globalIdx.y == domainDim.y - 1 ||
                            globalIdx.z == 0 || globalIdx.z == domainDim.z - 1) {
                            flagVal.classification = CellType::bounceBack;

                            if (globalIdx.y == domainDim.y - 1) {
                                flagVal.classification = CellType::movingWall;
                            }

                            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                                if (globalIdx.y == domainDim.y - 1) {
                                    popVal = -6. * Lattice::Memory::template getT<q>() * ulb *
                                             (Lattice::Memory::template getDirection<q>().v[0] * ulid.v[0] +
                                              Lattice::Memory::template getDirection<q>().v[1] * ulid.v[1] +
                                              Lattice::Memory::template getDirection<q>().v[2] * ulid.v[2]);
                                } else {
                                    popVal = 0;
                                }
                                fIn(gidx, q) = popVal;
                                fOut(gidx, q) = popVal;
                            });
                        } else {
                            flagVal.classification = CellType::bulk;
                            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                                fIn(gidx, q) = Lattice::Memory::template getT<q>();
                                fOut(gidx, q) = Lattice::Memory::template getT<q>();
                            });
                        }
                        cellInfoPartition(gidx, 0) = flagVal;
                    };
                });
            return container;
        }

        static auto
        setToEquilibrium(PopField&      fOutField,
                         CellTypeField& cellTypeField)
            -> Neon::set::Container
        {
            Neon::set::Container container = fOutField.getGrid().newContainer(
                "LBM_setToEquilibrium",
                [&](Neon::set::Loader& L) -> auto {
                    auto& fOut = L.load(fOutField, Neon::Pattern::MAP);
                    auto& cellInfoPartition = L.load(cellTypeField, Neon::Pattern::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename PopField::Idx& gidx) mutable {
                        {  // All pints are pre-set to bulk
                            CellType flagVal;
                            flagVal.classification = CellType::bulk;
                            cellInfoPartition(gidx, 0) = flagVal;
                        }

                        {  // All cells are pre-set to Equilibrium
                            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                                using M = typename Lattice::template RegisterMapper<q>;
                                fOut(gidx, M::fwdMemQ) = Lattice::Registers::template getT<M::fwdRegQ>();
                            });
                        }
                    };
                });
            return container;
        }
    };
};