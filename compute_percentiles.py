from sea_breeze.sea_breeze_filters import percentile
from dask.distributed import Client
import pandas as pd
import numpy as np
from sea_breeze.utils import load_diagnostics

if __name__ == "__main__":

    # Set up Dask client
    client = Client()

    # Set up the parameters
    #fields = ["F","Fc","sbi","fuzzy"]
    fields = ["F","sbi","fuzzy"]
    p = 99.5
    #models = ["era5","barra_r","barra_c_smooth_s2","aus2200_smooth_s4"]
    models = ["barra_r","barra_c_smooth_s2"]

    # Create a DataFrame to store the results
    vals_df = pd.DataFrame(index=models,columns=fields)

    # Loop through all combinations of fields and models
    # and calculate the percentiles
    for field in fields:
        for model in models:
            print(field,model)
            # If the model is "barra_r" or "barra_c_smooth_s2" and the field is "sbi", skip the calculation
            # because the data is not available for these models
            if (model in ["barra_r","barra_c_smooth_s2"]) and (field == "sbi"):
                pass
            else:
                # Load the data and calculate the percentile
                da = load_diagnostics(field,model)
                vals_df.at[model,field] = np.array(percentile(da,p))[0]
    
    # Save the results to a CSV file
    vals_df.to_csv("/g/data/ng72/ab4502/sea_breeze_detection/percentiles/percentiles_v2_"+str(p)+"_2013_2018.csv")

    # Close the Dask client
    client.close()