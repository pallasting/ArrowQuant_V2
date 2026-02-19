"""
MemoryVisualizer 单元测试
"""

import pytest
import tempfile
import os
import numpy as np

from llm_compression import MemoryVisualizer, MemoryPrimitive


@pytest.fixture
def memory_network():
    """创建测试记忆网络"""
    mem1 = MemoryPrimitive("mem_1", "content1", np.random.randn(384))
    mem2 = MemoryPrimitive("mem_2", "content2", np.random.randn(384))
    mem3 = MemoryPrimitive("mem_3", "content3", np.random.randn(384))
    
    # 添加连接
    mem1.add_connection("mem_2", 0.8)
    mem2.add_connection("mem_1", 0.8)
    mem2.add_connection("mem_3", 0.6)
    mem3.add_connection("mem_2", 0.6)
    
    # 设置激活
    mem1.activate(0.9)
    mem2.activate(0.7)
    mem3.activate(0.5)
    
    return {
        "mem_1": mem1,
        "mem_2": mem2,
        "mem_3": mem3
    }


@pytest.fixture
def visualizer(memory_network):
    """创建MemoryVisualizer实例"""
    return MemoryVisualizer(memory_network)


def test_initialization(visualizer, memory_network):
    """测试初始化"""
    assert visualizer.memory_network == memory_network


def test_generate_network_graph(visualizer):
    """测试生成网络图数据"""
    graph_data = visualizer.generate_network_graph()
    
    assert "nodes" in graph_data
    assert "edges" in graph_data
    assert "stats" in graph_data
    
    assert len(graph_data["nodes"]) == 3
    assert len(graph_data["edges"]) == 4  # 4条边（双向）


def test_node_structure(visualizer):
    """测试节点结构"""
    graph_data = visualizer.generate_network_graph()
    node = graph_data["nodes"][0]
    
    assert "id" in node
    assert "label" in node
    assert "access_count" in node
    assert "success_rate" in node
    assert "activation" in node
    assert "size" in node


def test_edge_structure(visualizer):
    """测试边结构"""
    graph_data = visualizer.generate_network_graph()
    edge = graph_data["edges"][0]
    
    assert "source" in edge
    assert "target" in edge
    assert "strength" in edge
    assert "width" in edge


def test_network_stats(visualizer):
    """测试网络统计"""
    graph_data = visualizer.generate_network_graph()
    stats = graph_data["stats"]
    
    assert stats["total_nodes"] == 3
    assert stats["total_edges"] == 2  # 双向连接算1条
    assert stats["avg_connections"] > 0
    assert stats["avg_activation"] > 0


def test_empty_network():
    """测试空网络"""
    visualizer = MemoryVisualizer({})
    graph_data = visualizer.generate_network_graph()
    
    assert len(graph_data["nodes"]) == 0
    assert len(graph_data["edges"]) == 0
    assert graph_data["stats"]["total_nodes"] == 0


def test_export_html(visualizer):
    """测试导出HTML"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        output_path = f.name
    
    try:
        visualizer.export_html(output_path, title="Test Visualization")
        
        # 验证文件存在
        assert os.path.exists(output_path)
        
        # 验证内容
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test Visualization" in content
            assert "d3.v7.min.js" in content  # 更精确的检查
            assert "mem_" in content  # 至少有一个记忆节点
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_html_contains_stats(visualizer):
    """测试HTML包含统计信息"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        output_path = f.name
    
    try:
        visualizer.export_html(output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Nodes:" in content
            assert "Edges:" in content
            assert "Avg Connections:" in content
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_node_size_scaling(visualizer):
    """测试节点大小缩放"""
    graph_data = visualizer.generate_network_graph()
    
    # 节点大小应该基于access_count
    for node in graph_data["nodes"]:
        expected_size = 10 + node["access_count"] * 2
        assert node["size"] == expected_size


def test_edge_width_scaling(visualizer):
    """测试边宽度缩放"""
    graph_data = visualizer.generate_network_graph()
    
    # 边宽度应该基于strength
    for edge in graph_data["edges"]:
        expected_width = edge["strength"] * 5
        assert edge["width"] == expected_width


def test_label_truncation(visualizer):
    """测试标签截断"""
    graph_data = visualizer.generate_network_graph()
    
    # 标签应该被截断到10个字符
    for node in graph_data["nodes"]:
        assert len(node["label"]) <= 10
