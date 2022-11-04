#include <list>
#include "Neon/skeleton/internal/MultiXpuGraph.h"

namespace Neon::skeleton::internal {

void MultiXpuGraph::init(Neon::Backend&                           bk,
                         const std::vector<Neon::set::Container>& operations,
                         std::string                              name,
                         Options                                  options)
{
    mGraph() = Neon::set::container::Graph(bk);
    parse(bk.devSet().setCardinality(),
          std::forward<const std::vector<Neon::set::Container>&&>(operations));
    mGraph().removeRedundantDependencies();

    h_io2Dot("t0_" + name + ".dot", "i");
    optimizations(options);
    h_io2Dot("t1_" + name + ".dot", "i");
    addCommunications(options);
    mGraph().removeRedundantDependencies();

    checkCoherency();
    h_io2Dot("t2_" + name + ".dot", "i");
    this->computeScheduling();
    h_io2Dot("schedulign" + name + ".dot", "i");
}

void MultiXpuGraph::parse(int                                       setCardinalty,
                          const std::vector<Neon::set::Container>&& operations)
{
    m_setCardinality() = setCardinalty;

    for (auto&& k : operations) {
        helpParseNewContainer(k);
    }
}

void MultiXpuGraph::
    helpParseNewContainer(const Neon::set::Container& inContainer)
{

    // Register and retrieve the id for the new container
    // add a node
    Neon::set::container::GraphInfo::NodeUid graphNodeUid;
    graphNodeUid = helpAddNewContainerToGraph(inContainer);

    // Parsing all the data toke used by the kernel container
    std::vector<Neon::set::dataDependency::Token> tokens;
    tokens = helpParseContainer(mGraph().helpGetGraphNode(graphNodeUid).getContainer());

    // Tokens are based on the multi-GPU data loaded by Containers
    for (auto& token : tokens) {
        // update the dependency state machine with the new token.
        // newDependencies are the detected dependencies

        auto newDependencies = m_dataRecords().updateStatus(graphNodeUid,
                                                            token.access(),
                                                            token.uid());

        for (auto& dep : newDependencies) {
            const auto& n0 = mGraph().helpGetGraphNode(dep.t0());
            const auto& n1 = mGraph().helpGetGraphNode(dep.t1());
            mGraph().addDependency(n0, n1, token);
        }
    }
}

auto MultiXpuGraph::
    helpParseContainer(Neon::set::Container& container)
        -> std::vector<Neon::set::dataDependency::Token>
{
    auto& kcInterface = container.getContainerInterface();
    auto& tokens = kcInterface.parse();
    return tokens;
}

auto MultiXpuGraph::h_io2Dot([[maybe_unused]] const std::string& fname,
                             [[maybe_unused]] const std::string& graphName) -> void
{
    io2Dot(fname, graphName, true);
}

auto MultiXpuGraph::
    io2Dot(const std::string& fname,
           const std::string& graphName,
           bool               debug) -> void
{
    mGraph().ioToDot(fname, graphName, debug);
}

auto MultiXpuGraph::
    io2DotOriginalApp(const std::string& fname,
                      const std::string& graphName,
                      bool               debug) -> void
{
    mGraph().ioToDot(fname, graphName, debug);
}

auto MultiXpuGraph::helpAddNewContainerToGraph(const Neon::set::Container& container) -> Neon::set::container::GraphInfo::NodeUid
{
    const auto&                              graphNode = mGraph().addNode(container);
    Neon::set::container::GraphInfo::NodeUid uid;
    uid = graphNode.getGraphData().getUid();
    return uid;
}


auto MultiXpuGraph::optimizations(const Neon::skeleton::Options& options) -> void
{
    switch (options.occ()) {
        case Neon::skeleton::Occ::none:
            return;
        case Neon::skeleton::Occ::standard:
            return optimizeStandardOCC(options);
        case Neon::skeleton::Occ::extended:
            return optimizeExtendedOCC(options);
        case Neon::skeleton::Occ::twoWayExtended:
            return optimizeTwoWayExtendedOCC(options);
    }
}

auto MultiXpuGraph::optimizeStandardOCC(const Neon::skeleton::Options&) -> void
{
    std::vector<Neon::set::container::GraphData::Uid> stencilNodeUidList;
    mGraph().forEachNode([&](const Neon::set::container::GraphNode& graphNode) {
        const auto type =
            graphNode.getContainer().getContainerInterface().getContainerPatternType();
        if (Neon::set::ContainerPatternType::stencil == type) {
            Neon::set::container::GraphData::Uid nodeUid = graphNode.getGraphData().getUid();
            stencilNodeUidList.push_back(nodeUid);
        }
    });

    for (auto stencilNodeUid : stencilNodeUidList) {
        auto& boundary = mGraph().helpGetGraphNode(stencilNodeUid);
        auto& internal = mGraph().cloneNode(boundary);

        internal.getScheduling().setDataView(Neon::DataView::INTERNAL);
        boundary.getScheduling().setDataView(Neon::DataView::BOUNDARY);

        mGraph().addDependency(internal, boundary, Neon::GraphDependencyType::scheduling);
    }
}

auto MultiXpuGraph::optimizeExtendedOCC(const Neon::skeleton::Options&) -> void
{
    NEON_DEV_UNDER_CONSTRUCTION("");

    //
    //    // Detects all stencil nodes
    //    std::vector<size_t> potentialStencilNodes;
    //    m_graph().forEachVertex([&](size_t nodeId) {
    //        const auto& node = m_graph().getVertexProperty(nodeId);
    //        if (node.getCompute() == Neon::Compute::STENCIL) {
    //            if (node.getDataView() == Neon::DataView::STANDARD) {
    //                potentialStencilNodes.push_back(nodeId);
    //            }
    //        }
    //    });
    //    std::vector<size_t> targetStencilNodes;
    //
    //    for (auto& potentialStencilNode : potentialStencilNodes) {
    //        bool attachedToRoot = false;
    //        bool proceedByMap = false;
    //        m_graph().forEachInEdge(potentialStencilNode, [&](const std::pair<size_t, size_t>& edge) {
    //            if (edge.first == m_rootNodeId()) {
    //                attachedToRoot = true;
    //            }
    //            if (!m_graph().getVertexProperty(edge.first).isStencil()) {
    //                proceedByMap = true;
    //            }
    //        });
    //        if (!attachedToRoot && proceedByMap) {
    //            targetStencilNodes.push_back(potentialStencilNode);
    //        }
    //    }
    //    // HELPER //////////////////////////////////////////////////////////////////////////
    //    auto h_cloneKernelNodeNode = [&](size_t node, const Neon::DataView& dataView) -> size_t {
    //        const auto& toBeCloned = m_graph().getVertexProperty(node);
    //        auto        clone = toBeCloned.clone();
    //        clone.setDataView(dataView);
    //        m_graph().addVertex(clone.nodeId(), clone);
    //        return clone.nodeId();
    //    };
    //
    //    // HELPER //////////////////////////////////////////////////////////////////////////
    //    auto h_cloneOutgoingConnections = [&](std::set<std::pair<size_t, size_t>> outEdges, size_t clone) -> void {
    //        std::for_each(outEdges.begin(), outEdges.end(),
    //                      [&](const DiGraph::Edge& edge) {
    //                          auto clonedEdge = m_graph().getEdgeProperty(edge).clone();
    //                          m_graph().addEdge(clone, edge.second, clonedEdge);
    //                      });
    //    };
    //
    //    // HELPER //////////////////////////////////////////////////////////////////////////
    //    auto h_cloneIncomingConnections = [&](std::set<std::pair<size_t, size_t>> inEdges, size_t clone) -> void {
    //        std::for_each(inEdges.begin(), inEdges.end(),
    //                      [&](const DiGraph::Edge& edge) {
    //                          auto clonedEdge = m_graph().getEdgeProperty(edge).clone();
    //                          m_graph().addEdge(edge.first, clone, clonedEdge);
    //                      });
    //    };
    //
    //    // HELPER //////////////////////////////////////////////////////////////////////////
    //    auto h_cloningProcess = [&](size_t st1, std::vector<size_t> mt0s) -> void {
    //        /**
    //         * st1: stencil node
    //         * mt0s: set of map nodes at the previous level
    //         *
    //         * st1 stencil t1
    //         * mto are map at t0
    //         */
    //        this->h_io2Dot("A.dot", "A");
    //        // Cloning the stencil node
    //        size_t st1_internal = h_cloneKernelNodeNode(st1, Neon::DataView::INTERNAL);
    //        size_t st1_boundary = h_cloneKernelNodeNode(st1, Neon::DataView::BOUNDARY);
    //        m_schedulingGraph().addVertex(st1_internal);
    //        m_schedulingGraph().addVertex(st1_boundary);
    //        m_schedulingGraph().addEdge(st1_internal, st1_boundary);
    //
    //        this->h_io2Dot("B.dot", "B");
    //
    //        {
    //            auto inE = m_graph().inEdges(st1);
    //            h_cloneIncomingConnections(inE, st1_internal);
    //            h_cloneIncomingConnections(inE, st1_boundary);
    //        }
    //        {
    //            auto outE = m_graph().outEdges(st1);
    //            h_cloneOutgoingConnections(outE, st1_internal);
    //            h_cloneOutgoingConnections(outE, st1_boundary);
    //        }
    //        this->h_io2Dot("C.dot", "B");
    //
    //        for (auto& mt0 : mt0s) {
    //            // cloning map node into boundary and internal
    //            size_t mapInternal = h_cloneKernelNodeNode(mt0, Neon::DataView::INTERNAL);
    //            size_t mapBoundary = h_cloneKernelNodeNode(mt0, Neon::DataView::BOUNDARY);
    //            m_schedulingGraph().addVertex(mapInternal);
    //            m_schedulingGraph().addVertex(mapBoundary);
    //            m_schedulingGraph().addEdge(mapBoundary, mapInternal);
    //
    //            {
    //                auto inE = m_graph().inEdges(mt0);
    //                h_cloneIncomingConnections(inE, mapInternal);
    //                h_cloneIncomingConnections(inE, mapBoundary);
    //            }
    //            this->h_io2Dot("D.dot", "B");
    //
    //            {
    //                auto outE = m_graph().outEdges(mt0);
    //                h_cloneOutgoingConnections(outE, mapInternal);
    //                h_cloneOutgoingConnections(outE, mapBoundary);
    //            }
    //
    //            {  // the stencil arc from map_internal and stencil{internal/boundary} mast be set to map
    //                for (auto& st1Target : {st1_internal, st1_boundary}) {
    //                    auto& edge = m_graph().getEdgeProperty({mapInternal, st1Target});
    //                    auto& infos = edge.infoMutable();
    //                    for (auto& info : infos) {
    //                        info.flag_discardStencilDep = true;
    //                    }
    //                }
    //            }
    //
    //            this->h_io2Dot("E.dot", "B");
    //            //            //m_graph().removeEdge({internal, st1_boundary});
    //            //            //m_graph().removeEdge({boundary, st1_internal});
    //            this->h_io2Dot("F.dot", "B");
    //        }
    //        m_graph().removeVertex(st1);
    //        this->h_io2Dot("G.dot", "B");
    //        for (auto& mt0 : mt0s) {
    //            m_graph().removeVertex(mt0);
    //        }
    //        this->h_io2Dot("H.dot", "B");
    //    };
    //
    //    ////////////////////////////////////////////////////////////////////////////
    //    auto node2Level = h_getBFSIndexes();
    //    for (auto&& stencilNode : targetStencilNodes) {
    //        int                 stencilNodeLevel = node2Level[stencilNode];
    //        std::vector<size_t> toClone;
    //        auto                inNodes = m_graph().inNeighbors(stencilNode);
    //        for (const auto& inNode : inNodes) {
    //            if (node2Level[inNode] == stencilNodeLevel - 1) {
    //                toClone.push_back(inNode);
    //            }
    //        }
    //        h_cloningProcess(stencilNode, toClone);
    //    }
}

auto MultiXpuGraph::optimizeTwoWayExtendedOCC(const Neon::skeleton::Options&) -> void
{
    NEON_DEV_UNDER_CONSTRUCTION("");

    //
    //    // Detects all stencil nodes
    //    std::vector<size_t> potentialStencilNodes;
    //    m_graph().forEachVertex([&](size_t nodeId) {
    //        const auto& node = m_graph().getVertexProperty(nodeId);
    //        if (node.getCompute() == Neon::Compute::STENCIL) {
    //            if (node.getDataView() == Neon::DataView::STANDARD) {
    //                potentialStencilNodes.push_back(nodeId);
    //            }
    //        }
    //    });
    //    std::vector<size_t> targetStencilNodes;
    //
    //    for (auto& potentialStencilNode : potentialStencilNodes) {
    //        bool attachedToRoot = false;
    //        bool attachedToFinal = false;
    //        bool proceedByMap = false;
    //        bool followedByMapOrDot = false;
    //
    //        m_graph().forEachInEdge(potentialStencilNode, [&](const std::pair<size_t, size_t>& edge) {
    //            if (edge.first == m_rootNodeId()) {
    //                attachedToRoot = true;
    //            }
    //            // only MAP front kernels can be splitted
    //            if (m_graph().getVertexProperty(edge.first).isMap()) {
    //                proceedByMap = true;
    //            }
    //        });
    //        m_graph().forEachOutEdge(potentialStencilNode, [&](const std::pair<size_t, size_t>& edge) {
    //            if (edge.second == m_finalNodeId()) {
    //                attachedToFinal = true;
    //            }
    //
    //            // only MAP or DOT (-> not STENCIL) back kernels can be splitted
    //            if (!m_graph().getVertexProperty(edge.second).isStencil()) {
    //                followedByMapOrDot = true;
    //            }
    //        });
    //        if (!attachedToRoot && !attachedToFinal && proceedByMap && followedByMapOrDot) {
    //            targetStencilNodes.push_back(potentialStencilNode);
    //        }
    //    }
    //    // HELPER //////////////////////////////////////////////////////////////////////////
    //    auto h_cloneKernelNodeNode = [&](size_t node, const Neon::DataView& dataView) -> size_t {
    //        const auto& toBeCloned = m_graph().getVertexProperty(node);
    //        auto        clone = toBeCloned.clone();
    //        clone.setDataView(dataView);
    //        m_graph().addVertex(clone.nodeId(), clone);
    //        return clone.nodeId();
    //    };
    //
    //    // HELPER //////////////////////////////////////////////////////////////////////////
    //    auto h_cloneOutgoingConnections = [&](std::set<std::pair<size_t, size_t>> outEdges, size_t clone) -> void {
    //        std::for_each(outEdges.begin(), outEdges.end(),
    //                      [&](const DiGraph::Edge& edge) {
    //                          auto clonedEdge = m_graph().getEdgeProperty(edge).clone();
    //                          m_graph().addEdge(clone, edge.second, clonedEdge);
    //                      });
    //    };
    //
    //    // HELPER //////////////////////////////////////////////////////////////////////////
    //    auto h_cloneIncomingConnections = [&](std::set<std::pair<size_t, size_t>> inEdges, size_t clone) -> void {
    //        std::for_each(inEdges.begin(), inEdges.end(),
    //                      [&](const DiGraph::Edge& edge) {
    //                          auto clonedEdge = m_graph().getEdgeProperty(edge).clone();
    //                          m_graph().addEdge(edge.first, clone, clonedEdge);
    //                      });
    //    };
    //
    //    // HELPER //////////////////////////////////////////////////////////////////////////
    //    auto h_cloningProcess = [&](const std::set<size_t>& frontNodes,
    //                                size_t                  StencilNode,
    //                                const std::set<size_t>& backNodes) -> void {
    //        std::set<size_t> backNodesInternal;
    //        std::set<size_t> backNodesBoundary;
    //        std::set<size_t> frontMapInternal;
    //
    //        /**
    //         * st1: stencil node
    //         * mt0s: set of map nodes at the previous level
    //         *
    //         * st1 stencil t1
    //         * mto are map at t0
    //         */
    //        this->h_io2Dot("A.dot", "A");
    //
    //        // Cloning from the back toward the front
    //        for (auto& mapBack : backNodes) {
    //            // cloning map node into boundary and internal
    //            size_t mapInternal = h_cloneKernelNodeNode(mapBack, Neon::DataView::INTERNAL);
    //            size_t mapBoundary = h_cloneKernelNodeNode(mapBack, Neon::DataView::BOUNDARY);
    //
    //            backNodesInternal.insert(mapInternal);
    //            backNodesBoundary.insert(mapBoundary);
    //
    //            m_schedulingGraph().addVertex(mapInternal);
    //            m_schedulingGraph().addVertex(mapBoundary);
    //
    //            const auto& metaNode = m_graph().getVertexProperty(mapBack);
    //            if (metaNode.isReduce()) {
    //                m_graph().addEdge(mapInternal, mapBoundary);
    //            } else {
    //                m_schedulingGraph().addEdge(mapInternal, mapBoundary);
    //            }
    //
    //
    //            {
    //                auto inE = m_graph().inEdges(mapBack);
    //                h_cloneIncomingConnections(inE, mapInternal);
    //                h_cloneIncomingConnections(inE, mapBoundary);
    //            }
    //            this->h_io2Dot("D.cloning.back.dot", "B");
    //
    //            {
    //                auto outE = m_graph().outEdges(mapBack);
    //                h_cloneOutgoingConnections(outE, mapInternal);
    //                h_cloneOutgoingConnections(outE, mapBoundary);
    //            }
    //
    //            this->h_io2Dot("E.cloning.back.dot", "B");
    //        }
    //
    //        for (auto& mapBack : backNodes) {
    //            m_graph().removeVertex(mapBack);
    //        }
    //
    //
    //        for (auto& mapFront : frontNodes) {
    //            // cloning map node into boundary and internal
    //            size_t mapBoundary = h_cloneKernelNodeNode(mapFront, Neon::DataView::BOUNDARY);
    //            size_t mapInternal = h_cloneKernelNodeNode(mapFront, Neon::DataView::INTERNAL);
    //
    //            frontMapInternal.insert(mapInternal);
    //
    //            m_schedulingGraph().addVertex(mapInternal);
    //            m_schedulingGraph().addVertex(mapBoundary);
    //            m_schedulingGraph().addEdge(mapBoundary, mapInternal);
    //
    //            {
    //                auto inE = m_graph().inEdges(mapFront);
    //                h_cloneIncomingConnections(inE, mapInternal);
    //                h_cloneIncomingConnections(inE, mapBoundary);
    //            }
    //            this->h_io2Dot("D.cloning.front.dot", "B");
    //
    //            {
    //                auto outE = m_graph().outEdges(mapFront);
    //                h_cloneOutgoingConnections(outE, mapInternal);
    //                h_cloneOutgoingConnections(outE, mapBoundary);
    //            }
    //
    //            this->h_io2Dot("E.cloning.front.dot", "B");
    //        }
    //        for (auto& mapFront : frontNodes) {
    //            m_graph().removeVertex(mapFront);
    //        }
    //
    //        // Cloning the stencil node
    //        size_t st1_internal = h_cloneKernelNodeNode(StencilNode, Neon::DataView::INTERNAL);
    //        size_t st1_boundary = h_cloneKernelNodeNode(StencilNode, Neon::DataView::BOUNDARY);
    //        m_schedulingGraph().addVertex(st1_internal);
    //        m_schedulingGraph().addVertex(st1_boundary);
    //        m_schedulingGraph().addEdge(st1_internal, st1_boundary);
    //
    //        this->h_io2Dot("B.dot", "B");
    //
    //        {
    //            auto inE = m_graph().inEdges(StencilNode);
    //            h_cloneIncomingConnections(inE, st1_internal);
    //            h_cloneIncomingConnections(inE, st1_boundary);
    //        }
    //        {
    //            auto outE = m_graph().outEdges(StencilNode);
    //            h_cloneOutgoingConnections(outE, st1_internal);
    //            h_cloneOutgoingConnections(outE, st1_boundary);
    //        }
    //
    //        for (auto& backNodeIternal : backNodesInternal) {
    //            m_schedulingGraph().addVertex(backNodeIternal);
    //            m_schedulingGraph().addVertex(st1_boundary);
    //            m_schedulingGraph().addEdge(backNodeIternal, st1_boundary);
    //        }
    //        this->h_io2Dot("C.dot", "B");
    //
    //        m_graph().removeVertex(StencilNode);
    //
    //
    //        for (const auto& node : backNodesBoundary) {
    //            m_graph().removeEdge(st1_internal, node);
    //        }
    //        for (const auto& node : backNodesInternal) {
    //            m_graph().removeEdge(st1_boundary, node);
    //        }
    //        {  // the stencil arc from map_internal and stencil{internal/boundary} mast be set to map
    //            for (auto& st1Target : {st1_internal, st1_boundary}) {
    //                for (auto& mapInternal : frontMapInternal) {
    //                    auto& edge = m_graph().getEdgeProperty({mapInternal, st1Target});
    //                    auto& infos = edge.infoMutable();
    //                    for (auto& info : infos) {
    //                        info.flag_discardStencilDep = true;
    //                    }
    //                }
    //            }
    //        }
    //
    //
    //        this->h_io2Dot("HHH.dot", "B");
    //    };
    //
    //    ////////////////////////////////////////////////////////////////////////////
    //    auto node2Level = h_getBFSIndexes();
    //    for (auto&& stencilNode : targetStencilNodes) {
    //        int              stencilNodeLevel = node2Level[stencilNode];
    //        std::set<size_t> frontToClone;
    //        std::set<size_t> backToClone;
    //
    //        auto inNodes = m_graph().inNeighbors(stencilNode);
    //        for (const auto& inNode : inNodes) {
    //            if (node2Level[inNode] == stencilNodeLevel - 1) {
    //                frontToClone.insert(inNode);
    //            }
    //        }
    //
    //        auto outNodes = m_graph().outNeighbors(stencilNode);
    //        for (const auto& outNode : outNodes) {
    //            if (outNode != m_finalNodeId()) {
    //                backToClone.insert(outNode);
    //            }
    //        }
    //        h_cloningProcess(frontToClone, stencilNode, backToClone);
    //    }
}


auto MultiXpuGraph::addCommunications(const Neon::skeleton::Options& skeletonOptions) -> void
{

    mGraph().ioToDot("before-comm", "appbeforecomm", true);

    if (m_setCardinality() == 1) {
        return;
    }

    // List all dependencies related to stencil operations
    std::vector<const Neon::set::container::GraphDependency*> stencilTypeDependencies;
    mGraph().forEachDependency([&](const Neon::set::container::GraphDependency& dep) {
        if (dep.getType() != Neon::GraphDependencyType::scheduling) {
            if (dep.hasStencilDependency()) {
                auto dw = dep.getDestinationNode(mGraph()).getScheduling().getDataView();
                if (Neon::DataView::INTERNAL != dw) {
                    stencilTypeDependencies.push_back(&dep);
                }
            }
        }
    });

    std::vector<const Neon::set::container::GraphDependency*> toBeRemoved;


    for (auto depPtr : stencilTypeDependencies) {
        const auto& dep = *depPtr;
        const auto  rawEdge = dep.getRawEdge();
        auto        nodeA = mGraph().helpGetGraphNode(rawEdge.first);
        auto        nodeB = mGraph().helpGetGraphNode(rawEdge.second);

        const Neon::set::container::GraphDependency::Tokens& tokens = dep.getTokens();
        int                                                  numNewNodes = 0;

        for (const auto& token : tokens) {
            if (token.compute() == Neon::Compute::STENCIL) {
                auto container = token.getDataTransferContainer(skeletonOptions.transferMode());
                numNewNodes += mGraph().expandAndMerge(nodeA, container, nodeB, true);
            }
        }

        auto schedulingDepOfB = this->mGraph().getProceedingGraphNodes(nodeB, {GraphDependencyType::scheduling});
        for (auto nodePtr : schedulingDepOfB) {
            const auto& dependency = mGraph().getDependency(*nodePtr, nodeB);
            toBeRemoved.push_back(&dependency);
        }
    }

//    for (auto depPtr : stencilTypeDependencies) {
//        const auto& dep = *depPtr;
//        const auto  rawEdge = dep.getRawEdge();
//        auto        nodeA = mGraph().helpGetGraphNode(rawEdge.first);
//        auto        nodeB = mGraph().helpGetGraphNode(rawEdge.second);
//        auto        schedulingDepOfB = this->mGraph().getProceedingGraphNodes(nodeB, {GraphDependencyType::scheduling});
//        for (auto nodePtr : schedulingDepOfB) {
//            const auto& dependency = mGraph().getDependency(*nodePtr, nodeB);
//            mGraph().removeDependency(dependency);
//        }
//        mGraph().removeDependency(*depPtr);
//    }

    for (auto depPtr : toBeRemoved) {
        mGraph().removeDependency(*depPtr);
    }


    // Final Step: we have to update the scheduling hints


    mGraph().ioToDot("after-comm", "app_after_comm", true);


    //
    //    // Detects all stencil nodes
    //    std::vector<Neon::set::container::GraphInfo::NodeUid> stencilNodesUids;
    //
    //    mGraph().forEachNode([&](Neon::set::container::GraphInfo::NodeUid nodeUid) {
    //        const auto& node = mGraph().getGraphNode(nodeUid);
    //        auto        pattern = node.getContainer().getContainerInterface().getContainerPatternType();
    //        if (pattern == Neon::set::ContainerPatternType::stencil) {
    //            auto dw = node.getScheduling().getDataView();
    //            if (dw != Neon::DataView::INTERNAL) {
    //                stencilNodesUids.push_back(nodeUid);
    //            }
    //        }
    //    });
    //
    //    for (auto&& stencilNodeUid : stencilNodesUids) {
    //        const auto& stencilNode = mGraph().getGraphNode(stencilNodeUid);
    //
    //        const auto             inEdges = m_graph().inEdges(stencilNode);
    //        int                    detectedStencilEdges = 0;
    //        [[maybe_unused]] auto& nodeInfo = m_graph().getVertexProperty(stencilNode);
    //
    //
    //        // m_graph().forEachInEdge(stencilNode,
    //        // We are using a copy of the inEdges,
    //        // so we can modify the graph without any issue
    //        std::for_each(inEdges.begin(), inEdges.end(),
    //                      [&](const DiGraph::Edge& edge) {
    //                          bool  visited = false;
    //                          auto& edgeP = m_graph().getEdgeProperty(edge);
    //                          auto& targetProperty = edgeP.infoMutable();
    //
    //                          // remove_if only  move elements at the end of the vector
    //                          auto end = std::remove_if(targetProperty.begin(),
    //                                                    targetProperty.end(),
    //                                                    [&](Edge::Info& info) -> bool {
    //                                                        if (info.token.compute() == Neon::Compute::STENCIL && !info.flag_discardStencilDep) {
    //                                                            visited = true;
    //                                                            detectedStencilEdges++;
    //                                                            h_add_hUpdateSyncNodes(edge.first,
    //                                                                                   edge.second,
    //                                                                                   info,
    //                                                                                   options.transferMode());
    //
    //                                                            return true;
    //                                                        } else {
    //                                                            visited = true;
    //                                                            return false;
    //                                                        }
    //                                                    });
    //
    //                          targetProperty.erase(end, targetProperty.end());
    //                          if (detectedStencilEdges > 0 && m_graph().getEdgeProperty(edge).nDependencies() == 0) {
    //                              m_graph().removeEdge(edge);
    //                          }
    //                      });
    //
    //        if (detectedStencilEdges != 1) {
    //            NEON_THROW_UNSUPPORTED_OPTION("Only one stencil in field is supported for now");
    //        }
    //    }
    //    NEON_DEV_UNDER_CONSTRUCTION("");
}

auto MultiXpuGraph::checkCoherency() -> void
{
    //    // Detects all stencil nodes
    //    std::vector<size_t> stencilNodes;
    //    m_graph().forEachVertex([&](size_t nodeId) {
    //        const auto& node = m_graph().getVertexProperty(nodeId);
    //        if (node.getCompute() == Neon::Compute::STENCIL) {
    //            if (m_setCardinality() == 1) {
    //                m_graph().getVertexProperty(nodeId).setAsCoherent();
    //                return;
    //            }
    //
    //            if (node.getDataView() == Neon::DataView::INTERNAL) {
    //                m_graph().getVertexProperty(nodeId).setAsCoherent();
    //            } else {
    //                stencilNodes.push_back(nodeId);
    //            }
    //        }
    //    });
    //    for (auto&& stencilNode : stencilNodes) {
    //        bool isMemoryCoherent = true;
    //        m_graph().forEachInEdge(stencilNode, [&](const DiGraph::Edge& edge) -> void {
    //            const auto& e = m_graph().getEdgeProperty(edge);
    //            for (const auto& i : e.info()) {
    //                if (i.isStencil() && !i.isHu()) {
    //                    isMemoryCoherent = false;
    //                }
    //            }
    //        });
    //        if (isMemoryCoherent) {
    //            m_graph().getVertexProperty(stencilNode).setAsCoherent();
    //        }
    //    }
}

MultiXpuGraph::
    MultiXpuGraph()
{
    m_storage = std::make_shared<Storage>();
}

auto MultiXpuGraph::
    computeScheduling() -> void
{
    mGraph().runtimePreSet(Neon::Backend::mainStreamIdx);
}

auto MultiXpuGraph::
    execute()
        -> void
{
    this->mGraph().helpExecute(Neon::Backend::mainStreamIdx);
}
}  // namespace Neon::skeleton::internal