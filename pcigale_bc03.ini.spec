data_file = string()
parameters_file = string()
sed_modules = cigale_string_list()
analysis_method = string()
cores = integer(min=1)
bands = cigale_string_list()
properties = cigale_string_list()
additionalerror = float(min=0.0)
[sed_modules_params]
  [[sfh2exp]]
    tau_main = cigale_list()
    tau_burst = cigale_list()
    f_burst = cigale_list(minvalue=0., maxvalue=0.9999)
    age = cigale_list(dtype=int, minvalue=0.)
    burst_age = cigale_list(dtype=int, minvalue=1.)
    sfr_0 = cigale_list(minvalue=0.)
    normalise = boolean()
  [[bc03]]
    imf = cigale_list(dtype=int, options=0. & 1.)
    metallicity = cigale_list(options=0.0001 & 0.0004 & 0.004 & 0.008 & 0.02 & 0.05)
    separation_age = cigale_list(dtype=int, minvalue=0)
  [[redshifting]]
    redshift = cigale_list()
[analysis_params]
  variables = cigale_string_list()
  save_sed = boolean()
  blocks = integer(min=1)
