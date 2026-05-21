import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

def get_flux_in_phot_filter(wavs_x1d, flux_1d, band):
    # Load in the filter curve
    trans_curve = np.loadtxt(f'{band}.txt')
    wavs_curve = trans_curve[:, 0]  # Wavelengths 
    if wavs_curve[0] > 1e3: # Units are in Angstroms
        wavs_curve = wavs_curve / 1e4  # Convert to microns
    transmission = trans_curve[:, 1]  # Transmission values

    # Interpolate over NaNs in flux_1d before interpolating to wavs_curve
    flux_1d_clean = flux_1d.copy()
    nan_mask = np.isnan(flux_1d_clean)
    if np.any(nan_mask):
        flux_1d_clean[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), flux_1d_clean[~nan_mask])

    # Interpolate the spectrum curve to match the wavelengths of the transmission curve
    flux_interp = np.interp(wavs_curve, wavs_x1d, flux_1d_clean, left=0, right=0)
    
    # Calculate the flux in the filter band
    flux = np.trapz(flux_interp * transmission, wavs_curve) / np.trapz(transmission, wavs_curve)
    print(f'Flux in {band} filter: {flux:.3e} muJy')

    return flux

def define_phot_filters():
    ## Define the filter wavenelghts
    filters_lambda_eff = {
                            'cfht-u'      :       [0.3690, 0.0456],
                            'hsc-g'       :       [0.4851, 0.1194],
                            'hsc-r'       :       [0.6241, 0.1539],
                            'hsc-i'       :       [0.7716, 0.1476],
                            'hsc-z'       :       [0.8915, 0.0768],
                            'hsc-y'       :       [0.9801, 0.0797],
                            'uvista-y'    :       [1.0222, 0.0919],
                            'uvista-j'    :       [1.2555, 0.1712],
                            'uvista-h'    :       [1.6497, 0.2893],
                            'uvista-ks'   :       [2.1577, 0.2926],
                            'hst-f814w'   :       [0.8068, 0.1610],
                            'f115w'       :       [1.1622, 0.2646],
                            'f150w'       :       [1.5106, 0.3348],
                            'f277w'       :       [2.8001, 0.6999],
                            'f444w'       :       [4.4366, 1.1109],
                            'f770w'       :       [7.7108, 2.0735],
                            'sc-ib427'    :       [0.4264, 0.0207],
                            'sc-ia484'    :       [0.4851, 0.0228],
                            'sc-ib505'    :       [0.5064, 0.0231],
                            'sc-ia527'    :       [0.5262, 0.0242],
                            'sc-ib574'    :       [0.5766, 0.0272],
                            'sc-ia624'    :       [0.6234, 0.0301],
                            'sc-ia679'    :       [0.6783, 0.0336],
                            'sc-ib709'    :       [0.7075, 0.0316],
                            'sc-ia738'    :       [0.7363, 0.0323],
                            'sc-ia767'    :       [0.7687, 0.0364],
                            'sc-ib827'    :       [0.8246, 0.0344],
                            'sc-nb711'    :       [0.7120, 0.0073],
                            'sc-nb816'    :       [0.8150, 0.0120],
                            'hsc-nb0816'   :       [0.8168, 0.0110],
                            'hsc-nb0921'   :       [0.9205, 0.0133],
                            'hsc-nb1010'  :       [1.0100, 0.0094]
                            }
    return filters_lambda_eff

##################################################################
# Crossmatch with the COSMOS2025 photometry and normalise the flux 
##################################################################
with fits.open('COSMOSWeb_mastercatalog_v1.fits', mode = 'readonly') as f: # Master catalog with photometry from COSMOS2025, retrieved from https://cosmos2025.iap.fr/
    phot_cat = Table(f[1].data)
filters_lambda_eff = define_phot_filters()

# USER INPUTS - FILL THESE IN
ra, dec = None, None # RA and Dec of the target source, in degrees
flux_x1d = None # 1D flux array of the spectrum
wavs_x1d = None # 1D wavelength array of the spectrum

# Crossmatch RA&DEC with the photometric catalog to find the source and its photometry
target_coords = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
d2d = target_coords.separation(SkyCoord(phot_cat['ra'], phot_cat['dec'], unit=(u.deg, u.deg), frame='icrs')).arcsec
source_idx = np.where(d2d < 0.5)[0]
cat_target = phot_cat[source_idx[0]]

# Read in the the photometry for the target source
photometry = []
photometry_err = []
band_wave = []
band_wave_width = []
cols = []
phot_array_target = []
for band in filters_lambda_eff:
    photometry.append(cat_target[f'flux_model_{band}'])
    photometry_err.append(cat_target[f'flux_err-cal_model_{band}'])
    band_wave.append(filters_lambda_eff[band][0])
    band_wave_width.append(filters_lambda_eff[band][1])
photometry = np.asarray(photometry)
photometry_err = np.asarray(photometry_err)
band_wave = np.asarray(band_wave)
band_wave_width = np.asarray(band_wave_width)

# Only use photometric bands that are fully covered by the spectrum (i.e. the bandpass is fully within the wavelength range of the spectrum)
mask_good_bands = (band_wave - band_wave_width > np.min(wavs_x1d)) & (band_wave + band_wave_width < np.max(wavs_x1d))
print(f'Photometry bands used: {np.array(list(filters_lambda_eff.keys()))[mask_good_bands]}')

# Calculate the flux in each photometric band by integrating the spectrum over the filter curve
flux_spec = []
for band in np.array(list(filters_lambda_eff.keys()))[mask_good_bands]:
    flux_spec.append(get_flux_in_phot_filter(flux_x1d, band))
flux_spec = np.asarray(flux_spec)

# Calculate the photometric rescaling factor by comparing to the photometry
phot_fact = np.mean(flux_spec/(photometry[mask_good_bands]*1e-6))
flux_rescaled = flux_x1d / phot_fact