#!/usr/bin/env python3
"""
Success Rate Plotter for BC Policy Inference
Visualizes success rates across different cube configurations with detailed statistics using Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import argparse
from pathlib import Path

class SuccessRatePlotter:
    def __init__(self, csv_file):
        """Initialize with CSV file"""
        self.csv_file = csv_file
        self.data = None
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.csv_file)
            # Clean configuration names
            self.data['Configuration'] = self.data['Configuration'].str.replace('config_', '').str.replace('_random', '')
            print(f"✅ Loaded {len(self.data)} configurations from {self.csv_file}")
        except Exception as e:
            print(f"❌ Error loading {self.csv_file}: {e}")
            raise ValueError(f"Could not load data from {self.csv_file}")
    
    def create_stacked_bar_chart(self, save_path=None):
        """Create stacked bar chart showing successes vs failures"""
        fig = go.Figure()
        
        configs = self.data['Configuration']
        successes = self.data['Successes']
        failures = self.data['Attempts'] - self.data['Successes']
        success_rates = self.data['SuccessRate (%)']
        
        # Add success bars "green"
        fig.add_trace(go.Bar(
            x=configs,
            y=successes,
            name='Successes',
            marker_color='#2ECC71',
            opacity=0.8,
            hovertemplate='<b>%{x}</b><br>Successes: %{y}<br><extra></extra>'
        ))
        
        # Add failure bars (stacked)
        fig.add_trace(go.Bar(
            x=configs,
            y=failures,
            name='Failures',
            marker_color='#E74C3C',
            opacity=0.8,
            hovertemplate='<b>%{x}</b><br>Failures: %{y}<br><extra></extra>'
        ))
        
        # Add annotations for success rates and fractions
        annotations = []
        for (config, success, total, rate) in zip(configs, successes, self.data['Attempts'], success_rates):
            # Success rate on top
            annotations.append(
                dict(
                    x=config,
                    y=total + 0.1,  # Position above the bar
                    text=f'{rate:.1f}%',
                    showarrow=False,
                    font=dict(size=11, color='black', family='Computer Modern'),
                    xanchor='center',
                    yanchor='bottom'
                )
            )
            
            # Fraction inside success bar (only if there are successes)
            if success > 0:
                annotations.append(
                    dict(
                        x=config,
                        y=success/2,
                        text=f'{success}/{total}',
                        showarrow=False,
                        font=dict(size=9, color='black', family='Computer Modern', 
                                 style='normal', variant='normal'),
                        xanchor='center',
                        yanchor='middle'
                    )
                )
        
        # Calculate overall success rate
        overall_success_rate = (self.data['Successes'].sum() / self.data['Attempts'].sum()) * 100
        
        # Add overall success rate annotation
        annotations.append(
            dict(
                text=f'Overall Success Rate: {overall_success_rate:.1f}%',
                xref="paper", yref="paper",
                x=0.98, y=0.85,  # Position above legend (legend is typically around y=0.8)
                xanchor="right", yanchor="bottom",
                showarrow=False,
                font=dict(size=12, color="black", family='Computer Modern', weight='bold'),
                bgcolor="rgba(248, 248, 248, 0.9)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=6
            )
        )
        
        fig.update_layout(
            title=dict(
                text='BC Policy Success Rate Analysis',
                x=0.5,
                font=dict(size=16, family='Computer Modern')
            ),
            xaxis=dict(
                title=dict(text='Configuration', 
                          font=dict(size=12, family='Computer Modern')),
                tickfont=dict(size=10, family='Computer Modern')
            ),
            yaxis=dict(
                title=dict(text='Number of Trials', 
                          font=dict(size=12, family='Computer Modern')),
                tickfont=dict(size=10, family='Computer Modern')
            ),
            legend=dict(
                title=dict(
                    text=f'Results (Overall: {overall_success_rate:.1f}%)',  # Integrate into legend title
                    font=dict(size=11, family='Computer Modern')
                ),
                font=dict(size=10, family='Computer Modern'),
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="gray",
                borderwidth=1,
                x=1.02,
                y=0.95,
                xanchor="left",
                yanchor="top"
            ),
            barmode='stack',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            width=1000,    # Wider canvas
            showlegend=True,
            annotations=annotations[:-1],  # Remove the separate overall success rate annotation
            margin=dict(l=60, r=150, t=80, b=60)  # More space for legend
        )
        
        if save_path:
            self._save_plot(fig, save_path, "stacked_bar_chart")
        
        return fig
    
    def create_detailed_analysis(self, save_path=None):
        """Create detailed analysis with statistics"""
        # Create 2x2 subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Distribution of Success Rates',
                'Success Rates (Sorted)',
                'Overall Success/Failure Distribution',
                'Performance Statistics'
            ],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "domain"}, {"type": "table"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Success rate distribution (histogram)
        fig.add_trace(go.Histogram(
            x=self.data['SuccessRate (%)'],
            nbinsx=10,
            marker_color='#3498DB',
            opacity=0.7,
            name='Distribution',
            showlegend=False
        ), row=1, col=1)
        
        # Add mean line
        mean_rate = self.data['SuccessRate (%)'].mean()
        fig.add_vline(
            x=mean_rate,
            line_dash="dash",
            line_color="red",
            annotation_text=f'Mean: {mean_rate:.1f}%',
            row=1, col=1
        )
        
        # Plot 2: Success rate by configuration (sorted)
        df_sorted = self.data.sort_values('SuccessRate (%)')
        colors = ['#E74C3C' if x < 50 else '#F39C12' if x < 75 else '#2ECC71' 
                 for x in df_sorted['SuccessRate (%)']]
        
        fig.add_trace(go.Bar(
            x=df_sorted['Configuration'],
            y=df_sorted['SuccessRate (%)'],
            marker_color=colors,
            opacity=0.8,
            name='Success Rates',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Success Rate: %{y:.1f}%<br><extra></extra>'
        ), row=1, col=2)
        
        # Plot 3: Success vs failure pie chart
        successes = self.data['Successes'].sum()
        failures = (self.data['Attempts'] - self.data['Successes']).sum()
        
        fig.add_trace(go.Pie(
            labels=['Successes', 'Failures'],
            values=[successes, failures],
            marker_colors=['#2ECC71', '#E74C3C'],
            textinfo='label+percent+value',
            name='Overall Distribution',
            showlegend=False
        ), row=2, col=1)
        
        # Plot 4: Statistics table
        stats_data = [
            ['Total Configurations', len(self.data)],
            ['Total Trials', self.data['Attempts'].sum()],
            ['Total Successes', self.data['Successes'].sum()],
            ['Overall Success Rate', f"{(self.data['Successes'].sum() / self.data['Attempts'].sum()) * 100:.1f}%"],
            ['Mean Success Rate', f"{self.data['SuccessRate (%)'].mean():.1f}%"],
            ['Median Success Rate', f"{self.data['SuccessRate (%)'].median():.1f}%"],
            ['Best Performance', f"{self.data['SuccessRate (%)'].max():.1f}%"],
            ['Worst Performance', f"{self.data['SuccessRate (%)'].min():.1f}%"],
            ['Configs with 100%', sum(self.data['SuccessRate (%)'] == 100)],
            ['Configs with <50%', sum(self.data['SuccessRate (%)'] < 50)]
        ]
        
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='lightgray',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[[row[0] for row in stats_data], [row[1] for row in stats_data]],
                fill_color='white',
                align='left',
                font=dict(size=11),
                height=30
            )
        ), row=2, col=2)
        
        # Update layout
        fig.update_xaxes(title_text="Success Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Configurations", row=1, col=1)
        fig.update_xaxes(title_text="Configuration (sorted by success rate)", row=1, col=2)
        fig.update_yaxes(title_text="Success Rate (%)", row=1, col=2)
        
        fig.update_layout(
            title=dict(
                text='Detailed BC Policy Performance Analysis',
                x=0.5,
                font=dict(size=16, family='Computer Modern')
            ),
            template='plotly_white',
            height=700,  # More compact for thesis
            width=1000,  # Better for thesis page width
            font=dict(family='Computer Modern', size=10)  # Global font setting
        )
        
        # Update all axis labels with Times New Roman
        fig.update_xaxes(
            title_font=dict(size=11, family='Computer Modern'),
            tickfont=dict(size=9, family='Computer Modern')
        )
        fig.update_yaxes(
            title_font=dict(size=11, family='Computer Modern'),
            tickfont=dict(size=9, family='Computer Modern')
        )
        
        if save_path:
            self._save_plot(fig, save_path, "detailed_analysis")
        
        return fig
    
    def _save_plot(self, fig, save_path, plot_type):
        """Save plot with appropriate format"""
        save_path = Path(save_path)
        
        # Add plot type to filename if not already present
        if plot_type not in str(save_path):
            stem = save_path.stem
            suffix = save_path.suffix
            save_path = save_path.parent / f"{stem}_{plot_type}{suffix}"
        
        if save_path.suffix.lower() == '.svg':
            fig.write_image(save_path, format='svg', width=1200, height=800)
            print(f"✅ Vector graphic saved to {save_path}")
        elif save_path.suffix.lower() == '.html':
            fig.write_html(save_path)
            print(f"✅ Interactive HTML saved to {save_path}")
        else:
            # Default PNG with high DPI
            fig.write_image(save_path, format='png', width=1200, height=800, scale=2)
            print(f"✅ Plot saved to {save_path}")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("📊 SUCCESS RATE ANALYSIS SUMMARY".center(70))
        print("="*70)
        
        total_trials = self.data['Attempts'].sum()
        total_successes = self.data['Successes'].sum()
        overall_rate = (total_successes / total_trials) * 100
        
        print(f"\n🎯 Analysis Results:")
        print(f"   Total Configurations: {len(self.data)}")
        print(f"   Total Trials: {total_trials}")
        print(f"   Total Successes: {total_successes}")
        print(f"   Overall Success Rate: {overall_rate:.1f}%")
        print(f"   Best Config: {self.data.loc[self.data['SuccessRate (%)'].idxmax(), 'Configuration']} ({self.data['SuccessRate (%)'].max():.1f}%)")
        print(f"   Worst Config: {self.data.loc[self.data['SuccessRate (%)'].idxmin(), 'Configuration']} ({self.data['SuccessRate (%)'].min():.1f}%)")
    
    def plot_all(self, save_path=None, show_plots=True):
        """Generate all plots"""
        self.load_data()
        self.print_summary()
        
        figures = []
        
        # Stacked bar chart
        fig1 = self.create_stacked_bar_chart(save_path)
        figures.append(fig1)
        
        # Detailed analysis
        fig2 = self.create_detailed_analysis(save_path)
        figures.append(fig2)
        
        if show_plots:
            for fig in figures:
                fig.show()
        
        return figures

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="📊 Visualize BC Policy Success Rate Statistics with Plotly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python3 plot_success_rate_inference.py --csv config_success_rates.csv
    
    # Save as SVG
    python3 plot_success_rate_inference.py --csv config_success_rates.csv --output results.svg
    
    # Save as interactive HTML
    python3 plot_success_rate_inference.py --csv config_success_rates.csv --output results.html
    
    # Save as PNG
    python3 plot_success_rate_inference.py --csv config_success_rates.csv --output results.png
        """
    )
    
    parser.add_argument("--csv", type=str, required=True,
                       help="Path to CSV file containing success rate data")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for plots (.svg for vector, .html for interactive, .png for raster)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plots (useful for batch processing)")
    
    args = parser.parse_args()
    
    try:
        plotter = SuccessRatePlotter(args.csv)
        plotter.plot_all(save_path=args.output, show_plots=not args.no_show)
        
        print("\n✅ Success rate analysis completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())