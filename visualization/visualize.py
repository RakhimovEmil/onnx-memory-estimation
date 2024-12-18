from containers import MutableTensorInfo

from collections import defaultdict
import svgwrite
from typing import List

def visualize(model_name: str, strategy: str, tensor_infos: List[MutableTensorInfo], output_file="memory_usage.svg"):
    events = []

    for tensor in tensor_infos:
        size_mb = tensor.estimated_size / 1024**2
        events.append((tensor.lifetime_begin, tensor.tensor_name, size_mb))
        events.append((tensor.lifetime_end, tensor.tensor_name, -size_mb))
    events.sort()

    memory_usage = defaultdict(float)
    time_steps = []
    memory_snapshots = []

    for time, tensor_name, size_change in events:
        if not time_steps or time != time_steps[-1]:
            time_steps.append(time)
            memory_snapshots.append(memory_usage.copy())
        memory_usage[tensor_name] += size_change

    time_steps.append(events[-1][0])
    memory_snapshots.append(memory_usage.copy())

    svg_width = 1200
    svg_height = 800
    margin = 100

    max_memory = max(sum(snapshot.values()) for snapshot in memory_snapshots)
    max_time = max(time_steps)
    x_scale = (svg_width - 2 * margin) / max_time
    y_scale = (svg_height - 2 * margin) / max_memory

    dwg = svgwrite.Drawing(output_file, size=(svg_width, svg_height))
    dwg.add(dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill="white"))

    title_text = f"{model_name} | {strategy}"
    dwg.add(dwg.text(
        title_text,
        insert=(svg_width / 2, margin / 2),
        text_anchor="middle",
        font_size=16,
        font_weight="bold"
    ))

    dwg.add(dwg.line(
        start=(margin, svg_height - margin),
        end=(svg_width - margin, svg_height - margin),
        stroke="black",
        stroke_width=2
    ))
    dwg.add(dwg.line(
        start=(margin, svg_height - margin),
        end=(margin, margin),
        stroke="black",
        stroke_width=2
    ))

    dwg.add(dwg.text("The sequence number of the execution", insert=(svg_width / 2, svg_height - 50), text_anchor="middle", font_size=14))
    dwg.add(dwg.text("Memory (MB)", insert=(50, svg_height / 2), text_anchor="middle", font_size=14, transform="rotate(-90, 50, {0})".format(svg_height / 2)))

    grid_line_color = "#D3D3D3"
    num_ticks = 10

    for i in range(num_ticks + 1):
        y = svg_height - margin - y_scale * i * max_memory / num_ticks
        dwg.add(dwg.line(
            start=(margin, y),
            end=(svg_width - margin, y),
            stroke=grid_line_color,
            stroke_width=1,
            stroke_dasharray="5,5"
        ))

    for i in range(num_ticks + 1):
        x = margin + x_scale * i * max_time / num_ticks
        dwg.add(dwg.line(
            start=(x, margin),
            end=(x, svg_height - margin),
            stroke=grid_line_color,
            stroke_width=1,
            stroke_dasharray="5,5"
        ))

    bottom_values = [0] * (len(time_steps) - 1)
    tensor_colors = {tensor.tensor_name: f"rgb({(i*37)%255},{(i*73)%255},{(i*127)%255})" for i, tensor in enumerate(tensor_infos)}

    hover_text = dwg.text(
        "",
        insert=(margin, svg_height - margin / 4),
        text_anchor="start",
        font_size=12,
        fill="black",
        id="hover_info"
    )
    dwg.add(hover_text)

    for tensor in tensor_infos:
        tensor_name = tensor.tensor_name
        sizes = [
            snapshot[tensor_name] if tensor_name in snapshot else 0
            for snapshot in memory_snapshots[:-1]
        ]
        for i, size in enumerate(sizes):
            if size > 0:
                x = margin + x_scale * time_steps[i]
                y = svg_height - margin - y_scale * (bottom_values[i] + size)
                width = x_scale * (time_steps[i + 1] - time_steps[i])
                height = y_scale * size
                bottom_values[i] += size

                rect = dwg.rect(
                    insert=(x, y),
                    size=(width, height),
                    fill=tensor_colors[tensor_name],
                    onmouseover=f"document.getElementById('hover_info').textContent='Tensor: {tensor_name}, Size: {size:.2f} MB, Steps: {tensor.lifetime_begin}:{tensor.lifetime_end}';",
                    onmouseout="document.getElementById('hover_info').textContent='';"
                )
                dwg.add(rect)

    for i in range(num_ticks + 1):
        time = i * max_time / num_ticks
        x = margin + time * x_scale
        dwg.add(dwg.line(
            start=(x, svg_height - margin),
            end=(x, svg_height - margin + 5),
            stroke="black",
            stroke_width=1
        ))
        dwg.add(dwg.text(
            f"{int(time)}",
            insert=(x, svg_height - margin + 15),
            text_anchor="middle",
            font_size=10
        ))

    for i in range(num_ticks + 1):
        y_value = i * max_memory / num_ticks
        y = svg_height - margin - y_scale * y_value
        dwg.add(dwg.text(
            f"{int(y_value)}",
            insert=(margin - 5, y),
            text_anchor="end",
            font_size=10
        ))

    dwg.save()
    print(f"SVG-график сохранен в {output_file}")
