"""HTML export for bot statistics."""

from datetime import datetime
from typing import Any, Dict


def generate_stats_html(stats: Dict[str, Any]) -> str:
    """Generate an HTML table with bot usage statistics."""
    summary = stats["summary"]
    user_stats = stats["user_stats"]

    sorted_users = sorted(
        user_stats.items(), key=lambda x: x[1]["total_cost"], reverse=True
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bot Statistics - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #2563eb;
        }}
        .summary-card .label {{
            color: #666;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .cost {{ color: #16a34a; }}
        .operations {{ font-size: 12px; color: #666; }}
        .date {{ font-size: 12px; color: #888; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Bot Usage Statistics</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <div class="summary-card">
                <div class="value">{summary['total_users']}</div>
                <div class="label">Total Users</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary['total_requests']}</div>
                <div class="label">Total Requests</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary['total_minutes']:.1f}</div>
                <div class="label">Audio Minutes</div>
            </div>
            <div class="summary-card">
                <div class="value">${summary['total_cost']:.4f}</div>
                <div class="label">Total Cost</div>
            </div>
        </div>

        <h2>Per-User Statistics</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Username</th>
                    <th>Requests</th>
                    <th>Minutes</th>
                    <th>Cost</th>
                    <th>Operations</th>
                    <th>Activity Period</th>
                </tr>
            </thead>
            <tbody>
"""

    for i, (username, user_data) in enumerate(sorted_users, 1):
        operations_str = ", ".join(
            f"{op}({data['count']})" for op, data in user_data["operations"].items()
        )

        first = user_data.get("first_activity")
        last = user_data.get("last_activity")
        if first and last:
            activity_str = f"{first.strftime('%Y-%m-%d')} to {last.strftime('%Y-%m-%d')}"
        else:
            activity_str = "-"

        html += f"""                <tr>
                    <td>{i}</td>
                    <td>@{username}</td>
                    <td>{user_data['total_requests']}</td>
                    <td>{user_data['total_minutes']:.1f}</td>
                    <td class="cost">${user_data['total_cost']:.4f}</td>
                    <td class="operations">{operations_str}</td>
                    <td class="date">{activity_str}</td>
                </tr>
"""

    html += """            </tbody>
        </table>
    </div>
</body>
</html>
"""
    return html
