#include "Neon/set/container/graph/GraphNode.h"
#include "Neon/set/container/AnchorContainer.h"
#include "Neon/set/container/types/ContainerExecutionType.h"

namespace Neon::set::container {

GraphNode::GraphNode()
{
}

auto GraphNode::
    newBeginNode() -> GraphNode
{
    GraphNode node;
    node.mGraphNodeOrganization.setUid(GraphData::beginUid);
    node.mContainer = Neon::set::Container::factoryAnchor("Begin");
    return node;
}

auto GraphNode::
    newEndNode() -> GraphNode
{
    GraphNode node;
    node.mGraphNodeOrganization.setUid(GraphData::endUid);
    node.mContainer = Neon::set::Container::factoryAnchor("End");
    return node;
}

auto GraphNode::
    getGraphData() -> GraphData&
{
    return mGraphNodeOrganization;
}

auto GraphNode::
    getGraphData()
        const -> const GraphData&
{
    return mGraphNodeOrganization;
}

auto GraphNode::
    getScheduling() -> GraphNodeScheduling&
{
    return mGraphNodeScheduling;
}

auto GraphNode::
    getScheduling()
        const -> const GraphNodeScheduling&
{
    return mGraphNodeScheduling;
}

auto GraphNode::
    getContainer() -> Container&
{
    return mContainer;
}

auto GraphNode::
    getContainer() const -> const Container&
{
    return mContainer;
}

GraphNode::
    GraphNode(const Container& container, GraphData::Uid uid)
{
    mContainer = container;
    mGraphNodeOrganization.setUid(uid);
}

auto GraphNode::
    toString()
        const -> std::string
{
    return std::string();
}


auto GraphNode::
    helpGetDotProperties()
        const -> std::string
{
    if (getContainerOperationType() == Neon::set::ContainerOperationType::anchor) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::compute) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::communication) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::synchronization) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}
auto GraphNode::
    helpGetDotName()
        const -> std::string
{
    return getContainer().getName();
}
auto GraphNode::helpGetDotInfo()
    const -> std::string
{
    if (getContainerOperationType() == Neon::set::ContainerOperationType::anchor) {
        return {};
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::compute) {
        std::stringstream s;
        s << "Uid = " << getContainer().getUid();
        s << "DataView = " << Neon::DataViewUtil::toString(getScheduling().getDataView());
        return s.str();
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::communication) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::synchronization) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

auto GraphNode::
    getContainerOperationType() const -> Neon::set::ContainerOperationType
{
    return getContainer().getContainerInterface().getContainerOperationType();
}

auto GraphNode::
    getContainerpatternType() const -> Neon::set::ContainerPatternType
{
    return getContainer().getContainerInterface().getContainerPatternType();
}

auto GraphNode::
    getLabel(bool debug)
        const -> std::string
{
    auto containerOperationType = getContainerOperationType();

    auto printNodeInformation = [this]() {
        std::stringstream s;

        auto printPositiveOrNone = [](int val) {
            if (val >= 0) {
                return std::to_string(val);
            }
            return std::string("None");
        };
        auto printNonEptyListOrNone = [](const std::vector<int>& ids) {
            if (ids.size() == 0) {
                return std::string("None");
            }
            std::stringstream tmp;
            for (size_t i = 0; i < ids.size(); i++) {
                if (i != 0) {
                    tmp << " ";
                }
                tmp << ids[i];
            }
            return tmp.str();
        };

        s << "\\l - UID: " << getContainer().getUid();
        s << "\\l - Execution: " << getContainer().getContainerExecutionType();
        s << "\\lGraphData ";
        s << "\\l - Graph Node id: " << this->getGraphData().getUid();

        s << "\\lScheduling " << getContainer().getName();
        s << "\\l - DataView: " << getScheduling().getDataView();
        s << "\\l - Stream  : " << getScheduling().getStream();
        s << "\\l - Wait    : " << printNonEptyListOrNone(getScheduling().getDependentEvents());
        s << "\\l - Signal  : " << printPositiveOrNone(getScheduling().getEvent());
        s << "\\l ---- ";

        return s.str();
    };
    if (containerOperationType == Neon::set::ContainerOperationType::anchor) {
        if (this->getGraphData().beginUid == getGraphData().getUid()) {
            std::stringstream s;
            s << "Begin ";
            if (debug) {
                s << printNodeInformation();
            }
            return s.str();
        }
        if (this->getGraphData().endUid == getGraphData().getUid()) {
            std::stringstream s;
            s << "End ";
            if (debug) {
                s << printNodeInformation();
            }
            return s.str();
        }
        std::stringstream s;
        s << "Sporious Anchor ";
        if (debug) {
            s << printNodeInformation();
        }
        return s.str();
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }
    if (containerOperationType == Neon::set::ContainerOperationType::compute) {
        std::stringstream s;
        if (debug) {
            s << "Container " << getContainer().getName();
            s << printNodeInformation();
        } else {
            s << getContainer().getName();
        }
        return s.str();
    }
    if (containerOperationType == Neon::set::ContainerOperationType::communication) {
        std::stringstream s;
        s << "Halo Update "
             " - Name: "
          << getContainer().getName();
        s << " - UID: " << getContainer().getUid();
        return s.str();
    }
    if (containerOperationType == Neon::set::ContainerOperationType::synchronization) {
        std::stringstream s;
        s << "Sync "
             " - Name: "
          << getContainer().getName();
        s << " - UID: " << getContainer().getUid();
        return s.str();
    }
    if (containerOperationType == Neon::set::ContainerOperationType::graph) {
        std::stringstream s;
        if (debug) {
            s << "Graph " << getContainer().getName();
            s << printNodeInformation();
        } else {
            s << getContainer().getName();
        }
        return s.str();
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
    return std::string();
}


auto GraphNode::
    getLabelProperty()
        const -> std::string
{
    auto containerOperationType = getContainerOperationType();
    if (containerOperationType == Neon::set::ContainerOperationType::anchor) {
        if (this->getGraphData().beginUid == getGraphData().getUid()) {
            return R"(shape=doublecircle, style=filled, fillcolor="#d9d9d9", color="#6c6c6c")";
        }
        if (this->getGraphData().endUid == getGraphData().getUid()) {
            return R"(shape=doublecircle, style=filled, fillcolor="#d9d9d9", color="#6c6c6c")";
        }
        // NEON_THROW_UNSUPPORTED_OPERATION("");
        return R"(shape=doublecircle, style=filled, fillcolor="#d9d9d9", color="#6c6c6c")";
    }
    if (containerOperationType == Neon::set::ContainerOperationType::compute) {
        auto pattern = getContainerpatternType();
        if (pattern == Neon::set::ContainerPatternType::map) {
            return R"(style=filled, fillcolor="#b3de69", color="#5f861d")";
        }
        if (pattern == Neon::set::ContainerPatternType::stencil) {
            return R"(style=filled, fillcolor="#bebada", color="#4e4683")";
        }
        if (pattern == Neon::set::ContainerPatternType::reduction) {
            return R"(style=filled, fillcolor="#80b1d3", color="#2b5c7d")";
        }
        if (pattern == Neon::set::ContainerPatternType::complex) {
            return R"(style=filled, fillcolor="#ff33f6", color="#ff33f6")";
        }
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }

    if (containerOperationType == Neon::set::ContainerOperationType::communication) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }

    if (containerOperationType == Neon::set::ContainerOperationType::synchronization) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (containerOperationType == Neon::set::ContainerOperationType::graph) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#ff33f6", color="#ff33f6")";
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
    return std::string();
}

}  // namespace Neon::set::container
