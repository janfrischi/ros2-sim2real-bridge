import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse

class FilterMotorSimulation:
    """Simulation of cascaded EMA filter + Motor dynamics"""
    
    def __init__(self, tau1=0.5, tau2=0.8):
        self.tau1 = tau1  # EMA filter time constant
        self.tau2 = tau2  # Motor time constant
        
    def input_signal(self, t, signal_type='step', amplitude=1.0, frequency=1.0, slope=1.0):
        """Generate different input signals"""
        if signal_type == 'step':
            return amplitude * np.ones_like(t)
        elif signal_type == 'sinusoid':
            return amplitude * np.sin(2 * np.pi * frequency * t)
        elif signal_type == 'ramp':
            return slope * t
        elif signal_type == 'square':
            return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        elif signal_type == 'chirp':
            return amplitude * np.sin(2 * np.pi * frequency * t * (1 + t / 10))
        elif signal_type == 'impulse':
            pulse_width = 0.1
            return amplitude * ((t >= 0) & (t <= pulse_width)) / pulse_width
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
    
    def ode_system_with_intermediate(self, t, state, signal_type, amplitude, frequency, slope):
        """ODE system with intermediate signal z(t) exposed"""
        z, dz_dt, y, dy_dt = state
        
        # Input signal
        x = self.input_signal(t, signal_type, amplitude, frequency, slope)
        
        # EMA filter dynamics: tau1 * dz/dt + z = x
        dz_dt_new = (x - z) / self.tau1
        
        # Motor dynamics: tau2 * dy/dt + y = z
        dy_dt_new = (z - y) / self.tau2
        
        return [dz_dt_new, 0, dy_dt_new, 0]
    
    def simulate(self, t_span=(0, 10), n_points=1000, signal_type='step', 
                 amplitude=1.0, frequency=1.0, slope=1.0):
        """Run simulation and return results"""
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Simulate with intermediate signal
        y0 = [0.0, 0.0, 0.0, 0.0]  # [z, dz/dt, y, dy/dt]
        sol = solve_ivp(
            lambda t, state: self.ode_system_with_intermediate(
                t, state, signal_type, amplitude, frequency, slope
            ),
            t_span, y0, t_eval=t_eval, method='RK45'
        )
        
        # Extract signals
        z_signal = sol.y[0]  # Intermediate filtered signal
        y_signal = sol.y[2]  # Final output
        
        # Generate input signal for plotting
        x_signal = self.input_signal(t_eval, signal_type, amplitude, frequency, slope)
        
        return {
            'time': sol.t,
            'input': x_signal,
            'intermediate': z_signal,
            'output': y_signal
        }
    
    def plot_results(self, results, signal_type):
        """Plot simulation results using matplotlib"""
        
        plt.figure(figsize=(12, 8))
        
        # Plot all signals on the same plot
        plt.plot(results['time'], results['input'], 
                'k--', linewidth=3, label=f'Input x(t) - Reference Trajectory - {signal_type}')
        
        plt.plot(results['time'], results['intermediate'], 
                'b-', linewidth=2, label=f'Filtered z(t) - Intermediate Signal after EMA (τ₁={self.tau1}s)')
        
        plt.plot(results['time'], results['output'], 
                'r-', linewidth=2, label=f'Output y(t) - Final Output after Motor (τ₂={self.tau2}s)')
        
        # Formatting
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Signal Value', fontsize=12)
        plt.title(f'Filter + Motor Response to {signal_type.title()} Input', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc='best')
        
        # Make plot interactive if running in Jupyter or with interactive backend
        plt.tight_layout()
        plt.show()
        
        # Print analysis
        self.print_analysis(signal_type)
    
    def print_analysis(self, signal_type):
        """Enhanced system analysis with formatting"""
        print("\n" + "="*50)
        print(f"📊 SYSTEM ANALYSIS - {signal_type.upper()} INPUT")
        print("="*50)
        print(f"🔧 EMA Filter Time Constant (τ₁): {self.tau1:.3f} s")
        print(f"⚙️  Motor Time Constant (τ₂):     {self.tau2:.3f} s")
        print(f"🔗 Combined Time Constant:       {self.tau1 * self.tau2:.3f} s")
        print(f"📡 Natural Frequency:            {1/np.sqrt(self.tau1*self.tau2):.3f} rad/s")
        print(f"📉 Damping Ratio:                {(self.tau1+self.tau2)/(2*np.sqrt(self.tau1*self.tau2)):.3f}")
        
        # System classification
        damping_ratio = (self.tau1+self.tau2)/(2*np.sqrt(self.tau1*self.tau2))
        if damping_ratio > 1:
            system_type = "Overdamped 🐌"
        elif damping_ratio == 1:
            system_type = "Critically Damped ⚖️"
        else:
            system_type = "Underdamped 🌊"
        
        print(f"📈 System Type:                  {system_type}")
        
        # Add performance metrics for step response
        if signal_type == 'step':
            print(f"⏱️  Theoretical Settling Time:   {4 * max(self.tau1, self.tau2):.3f} s (4τ rule)")
        
        print("="*50)

def interactive_simulation():
    """Interactive simulation"""
    
    print("=== Filter + Motor Dynamics Simulation ===")
    print("Available: step, sinusoid, ramp, square, chirp, impulse")
    
    signal_type = input("Signal type (default: step): ").strip() or "step"
    tau1 = float(input("EMA time constant τ₁ (default: 0.5): ") or "0.5")
    tau2 = float(input("Motor time constant τ₂ (default: 0.8): ") or "0.8")
    amplitude = float(input("Amplitude (default: 1.0): ") or "1.0")
    
    frequency = 1.0
    slope = 1.0
    
    if signal_type in ['sinusoid', 'square', 'chirp']:
        frequency = float(input("Frequency Hz (default: 1.0): ") or "1.0")
    
    if signal_type == 'ramp':
        slope = float(input("Slope (default: 1.0): ") or "1.0")
    
    t_end = float(input("Simulation time (default: 10): ") or "10")
    
    # Run simulation
    sim = FilterMotorSimulation(tau1, tau2)
    results = sim.simulate(
        t_span=(0, t_end),
        signal_type=signal_type,
        amplitude=amplitude,
        frequency=frequency,
        slope=slope
    )
    
    # Plot results
    sim.plot_results(results, signal_type)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Filter + Motor Dynamics Simulation")
    parser.add_argument("--signal", default="step", 
                       choices=['step', 'sinusoid', 'ramp', 'square', 'chirp', 'impulse'])
    parser.add_argument("--tau1", type=float, default=0.5)
    parser.add_argument("--tau2", type=float, default=0.8)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--slope", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--interactive", action="store_true")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_simulation()
    else:
        sim = FilterMotorSimulation(args.tau1, args.tau2)
        results = sim.simulate(
            t_span=(0, args.time),
            signal_type=args.signal,
            amplitude=args.amplitude,
            frequency=args.frequency,
            slope=args.slope
        )
        sim.plot_results(results, args.signal)

if __name__ == "__main__":
    main()
