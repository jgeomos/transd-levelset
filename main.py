#!/usr/bin/env python3
"""
Main script for trans-dimensional geophysical inversion.

This script runs the trans-dimensional inversion workflow using parameters
from a configuration file (parfile). It performs Bayesian inversion of 
gravity/magnetic data using MCMC sampling with level set methods.

Usage:
    python main.py -p parfile_transd_synth1.txt -s 42
    python main.py -p parfile_transd_synth1.txt -s 123 --no-logging

Authors:
    Jérémie Giraud
    Vitaliy Ogarko

License: MIT
"""

__version__ = "0.1.0"
__email__ = "jeremie.giraud@uwa.edu.au"


import sys
import argparse
from pathlib import Path
import numpy as np
import shutil

# Home-made libs
import src.transd_runner as tr
import src.input_output.input_params as input_params
import src.input_output.output_manager as om
import src.utils.plot_utils as ptu


def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run trans-dimensional geophysical inversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'parfile',
        type=str,
        help='Path to parameter file (e.g., parfile_transd_synth1.txt)'
    )
    parser.add_argument(
        'seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no-logging',
        action='store_true',
        help='Disable logging entirely'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Display metrics plots after inversion completes'
    )
    
    args = parser.parse_args()
    
    # Validate parfile exists
    parfile_path = Path(args.parfile)
    if not parfile_path.exists():
        print(f"Error: Parameter file '{args.parfile}' not found")
        sys.exit(1)
    
    # Setup logging using the existing LogRun class
    if args.no_logging:
        log_run = om.LogRun(return_none=True)
    else:
        log_run = om.LogRun(
            log_name='log_file.log',  # TODO create this name dynamically using e.g. the run ID. 
            logger_name='transd_main',
            to_console=True,
            verbose=True
        )
    
    if log_run is not None:
        log_run.info("="*70)
        log_run.info("TRANS-DIMENSIONAL GEOPHYSICAL INVERSION")
        log_run.info("="*70)
        log_run.info(f"Parameter file: {parfile_path.absolute()}")
        log_run.info(f"Random seed: {args.seed}")
    else:
        print(f"Parameter file: {parfile_path.absolute()}")
        print(f"Random seed: {args.seed}")
    
    # Initialize random number generator
    rng_main = np.random.default_rng(args.seed)
    
    # Read input parameters
    if log_run is not None:
        log_run.info("\nReading input parameters...")
    par = input_params.read_input_parameters(str(parfile_path), log_run)
    
    # Copy parameter file to output directory for reproducibility
    output_path = Path(par.path_output)
    if output_path.exists():
        dest_parfile = output_path / parfile_path.name
        shutil.copy2(parfile_path, dest_parfile)
        if log_run is not None:
            log_run.info(f"Parameter file copied to: {dest_parfile}")
        
        # Save seed to text file for easy reference
        seed_file = output_path / 'seed.txt'
        seed_file.write_text(str(args.seed))
        if log_run is not None:
            log_run.info(f"Seed saved to: {seed_file}")
    
    # Initialize run configuration
    run_config = input_params.RunBaseParams(logging=True)
    
    # Run the inversion
    if log_run is not None:
        log_run.info("\n" + "="*70)
        log_run.info("STARTING INVERSION")
        log_run.info("="*70 + "\n")
    
    try:
        results = tr.run_transd(
            par=par,
            log_run=log_run,
            rng_main=rng_main,
            run_config=run_config,
            sensit=None,
            folder_transd=""
        )
        
        petrovals, birth_params, mvars, metrics, shpars, gpars, \
            phipert_config, geophy_data, sensit, spars, par = results
        
        if log_run is not None:
            log_run.info("\n" + "="*70)
            log_run.info("INVERSION COMPLETED SUCCESSFULLY")
            log_run.info("="*70)
            log_run.info(f"Output directory: {spars.path_output}")
        else:
            print(f"\nInversion completed. Output directory: {spars.path_output}")
        
        # Plot metrics if requested
        if args.plot:
            if log_run is not None:
                log_run.info("\nGenerating metrics plots...")
            
            # Generate the plot and save.
            plot_path = Path(spars.path_output) / 'metrics_plot.png'
            ptu.plot_metrics(metrics, run_config, data_misfit_lims=np.array([0, 6]), 
                             print_stats=True, show=False, save=True, plot_path=plot_path)
 
            if log_run is not None:
                log_run.info(f"Metrics plot saved to: {plot_path}")
            else:
                print(f"Metrics plot saved to: {plot_path}")

        # Saving metrics.
        om.save_metrics_summary(metrics, f"{spars.path_output}/metrics_summary.txt")
        om.save_metrics_data(metrics, f"{spars.path_output}/metrics_data.csv")
        
    except Exception as e:
        if log_run is not None:
            log_run.error(f"\nInversion failed with error: {e}")
            import traceback
            log_run.error(traceback.format_exc())
        else:
            print(f"\nInversion failed with error: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up logging
        if log_run is not None:
            log_run.close()


if __name__ == "__main__":
    main()
