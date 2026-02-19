"""
记忆网络可视化

使用NetworkX和Plotly生成交互式记忆网络可视化
"""

from typing import Dict, List, Optional
import json

try:
    import networkx as nx
except ImportError:
    nx = None

from .memory_primitive import MemoryPrimitive


class MemoryVisualizer:
    """记忆网络可视化器"""
    
    def __init__(self, memory_network: Dict[str, MemoryPrimitive]):
        self.memory_network = memory_network
        
        if nx is None:
            raise ImportError(
                "NetworkX is required for visualization. "
                "Install with: pip install networkx"
            )
    
    def generate_network_graph(self) -> Dict:
        """
        生成网络图数据（JSON格式）
        
        Returns:
            包含nodes和edges的字典
        """
        nodes = []
        edges = []
        
        for mem_id, memory in self.memory_network.items():
            # 节点数据
            node = {
                "id": mem_id,
                "label": mem_id[:10],  # 简短标签
                "access_count": memory.access_count,
                "success_rate": memory.get_success_rate(),
                "activation": memory.activation,
                "size": 10 + memory.access_count * 2  # 节点大小
            }
            nodes.append(node)
            
            # 边数据
            for target_id, strength in memory.connections.items():
                if target_id in self.memory_network:
                    edge = {
                        "source": mem_id,
                        "target": target_id,
                        "strength": strength,
                        "width": strength * 5  # 边宽度
                    }
                    edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self._get_network_stats()
        }
    
    def export_html(
        self,
        output_path: str,
        title: str = "Memory Network Visualization"
    ):
        """
        导出交互式HTML可视化
        
        Args:
            output_path: 输出文件路径
            title: 可视化标题
        """
        graph_data = self.generate_network_graph()
        
        html_content = self._generate_html_template(
            graph_data=graph_data,
            title=title
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _get_network_stats(self) -> Dict:
        """获取网络统计"""
        if not self.memory_network:
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "avg_connections": 0.0,
                "avg_activation": 0.0
            }
        
        total_connections = sum(
            len(mem.connections)
            for mem in self.memory_network.values()
        )
        
        total_activation = sum(
            mem.activation
            for mem in self.memory_network.values()
        )
        
        return {
            "total_nodes": len(self.memory_network),
            "total_edges": total_connections // 2,  # 双向连接
            "avg_connections": total_connections / len(self.memory_network),
            "avg_activation": total_activation / len(self.memory_network)
        }
    
    def _generate_html_template(
        self,
        graph_data: Dict,
        title: str
    ) -> str:
        """生成HTML模板"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        #container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-top: 0;
        }}
        #stats {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }}
        #graph {{
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .node {{
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .node-label {{
            font-size: 10px;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <div id="container">
        <h1>{title}</h1>
        <div id="stats">
            <strong>Network Statistics:</strong><br>
            Nodes: {graph_data['stats']['total_nodes']} | 
            Edges: {graph_data['stats']['total_edges']} | 
            Avg Connections: {graph_data['stats']['avg_connections']:.2f} | 
            Avg Activation: {graph_data['stats']['avg_activation']:.2f}
        </div>
        <svg id="graph" width="1160" height="600"></svg>
    </div>
    
    <script>
        const data = {json.dumps(graph_data)};
        
        const width = 1160;
        const height = 600;
        
        const svg = d3.select("#graph");
        
        // 创建力导向图
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.edges).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        // 绘制边
        const link = svg.append("g")
            .selectAll("line")
            .data(data.edges)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => d.width);
        
        // 绘制节点
        const node = svg.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => d.size)
            .attr("fill", d => {{
                const activation = d.activation;
                return `hsl(${{activation * 120}}, 70%, 50%)`;
            }})
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        // 节点标签
        const label = svg.append("g")
            .selectAll("text")
            .data(data.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.label);
        
        // 更新位置
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x + 12)
                .attr("y", d => d.y + 4);
        }});
        
        // 拖拽函数
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // 节点提示
        node.append("title")
            .text(d => `${{d.id}}\\nAccess: ${{d.access_count}}\\nSuccess: ${{(d.success_rate * 100).toFixed(1)}}%`);
    </script>
</body>
</html>"""
