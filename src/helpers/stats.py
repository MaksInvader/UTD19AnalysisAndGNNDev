import os
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import norm
import numpy as np

import pandas as pd
import os
import plotly.express as px

def plot_error_distributions_map(best_sensor_rmse, best_sensor_mae, best_sensor_accuracy, best_sensor_r2, best_sensor_variance, output_path, sensor_ids, geocoordinates):
    save_path = os.path.join(output_path, "stats")
    os.makedirs(save_path, exist_ok=True) # Ensure the directory exists

    metrics_to_plot = [
        ("Test Root Mean Square Error (RMSE)", "RMSE", best_sensor_rmse),
        ("Test Mean Absolute Error (MAE)", "MAE", best_sensor_mae),
        ("Test Accuracy (Acc)", "Acc", best_sensor_accuracy),
        ("Test Coefficient of Determination (R²)", "R²", best_sensor_r2),
        ("Test Variance (Var)", "Var", best_sensor_variance)
    ]

    # Prepare data for heatmap by aligning metrics with coordinates
    heatmap_data = geocoordinates.loc[sensor_ids].copy()
    
    for metric_label, metric_name, metric_values in metrics_to_plot:
        print(f"Generating {metric_name} stats...")
        
        # Add the metric values to the dataframe for easy plotting
        heatmap_data[metric_name] = metric_values

        # Plotly scatter mapbox
        fig = px.scatter_mapbox(heatmap_data, 
                                lat="lat", 
                                lon="long", 
                                color=metric_name, 
                                color_continuous_scale="viridis",
                                title=f"Heatmap of {metric_label}",
                                size_max=20, # Increased size for better visibility
                                zoom=11,     # Adjusted zoom for a better view
                                labels={"color": ""})

        # Set a consistent size for all markers
        fig.update_traces(marker={'size': 15})

        # MODIFIED: Changed mapbox_style and removed the access token
        fig.update_layout(
            mapbox_style="carto-positron", # Using a token-free open-source map style
            # mapbox_accesstoken="..." line has been removed
            title=dict(yanchor="top", y=0.97, xanchor="center", x=0.5),
            font_family="Arial, sans-serif", # Using a more common font
            font_color="#333333",
            title_font_size=24,
            font_size=16,
            legend=dict(
                orientation="v", 
                yanchor="middle", 
                y=0.5, 
                xanchor="left", 
                x=1.05
            )
        )

        fig.update_coloraxes(colorbar_tickfont_size=14)
        
        # Save the interactive map to an HTML file
        output_file = os.path.join(save_path, f'{metric_name.lower()}_heatmap.html')
        fig.write_html(output_file)
        print(f"Saved heatmap to {output_file}")




def plot_error_distributions(best_sensor_rmse, best_sensor_mae, best_sensor_accuracy, best_sensor_r2, best_sensor_variance, output_path):
    save_path = os.path.join(output_path, "stats")

    metrics_to_plot = [
        ("Test Root Mean Square Error (RMSE)", "RMSE", best_sensor_rmse),
        ("Test Mean Absolute Error (MAE)", "MAE", best_sensor_mae),
        ("Test Accuracy (Acc)", "Acc",best_sensor_accuracy),
        ("Test Coefficient of Determination (R²)", "R²", best_sensor_r2),        
        ("Test Variance (Var)", "Var", best_sensor_variance)
    ]

    for metric_label, metric_name, metric_values in metrics_to_plot:
        metric_values = sorted(metric_values)

        print(f"Generating {metric_name} stats...")

        fit_data = stats.norm.cdf(metric_values, np.mean(metric_values), np.std(metric_values))

        plt.title(f"Normal Distribution of the Min {metric_name} Values for All Sensors", fontweight="bold",
                  color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
        plt.yticks(fontweight="bold", color="#333333",
                   fontname='Times New Roman', fontsize=8)
        plt.xticks(fontweight="bold", color="#333333",
                   fontname='Times New Roman', fontsize=8)

        plt.plot(metric_values, fit_data, linewidth=1, marker=".", markersize=2, color="#ffd700", label=f"{metric_name}, σ = " +
                 str(round(np.std(metric_values), 2)) + ", µ = " + str(round(np.mean(metric_values), 2)),)
        plt.legend(loc='best', fontsize=8)

        plt.axvline(x=np.mean(metric_values), linewidth=1,
                    color='#ffd700', ls='--', label='axvline - full height')

        plt.xlabel(f'{metric_label}', fontweight="bold", color="#333333",
                   fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
        plt.ylabel('Frequency in %', fontweight="bold", color="#333333",
                   fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)

        plt.savefig(save_path+f'/{metric_name.lower()}_distribution.jpg', dpi=500)
        plt.close()





def plot_additional_errors(test_rmse, train_rmse, train_loss, alpha1, save_path):
    # plot additional errors for the model
    # train_rmse & test_rmse
    fig1 = plt.figure(figsize=(6, 4))
    plt.title(str("Training vs Test Root Mean Square Error (RMSE)"), fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y labels
    plt.xlabel('Epochs', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('RMSE', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(train_rmse, label="train_rmse", color="#ffd700")
    plt.plot(test_rmse, label="test_rmse", color="#333333")
    plt.legend(loc='best', fontsize=8)
    # save the figure
    plt.savefig(save_path+'/rmse.jpg', dpi=500)
    # plt.show()
    # close the figure
    plt.close()

    # train_loss
    fig1 = plt.figure(figsize=(6, 4))
    plt.title(str("Training Loss"), fontweight="bold", color="#333333",
              pad=10, fontname='Times New Roman', fontdict={'fontsize': 10})
    # x y labels
    plt.xlabel('Epochs', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('Loss', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(train_loss, label='train_loss', color="#333333")
    plt.yscale('log')
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/train_loss.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()

    # train_rmse
    fig1 = plt.figure(figsize=(6, 4))
    plt.title(str("Training Root Mean Square Error (RMSE)"), fontweight="bold",
              color="#333333", pad=10, fontname='Times New Roman', fontdict={'fontsize': 12})
    # x y labels
    plt.xlabel('Epochs', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    plt.ylabel('RMSE', fontweight="bold", color="#333333",
               fontname='Times New Roman', fontdict={'fontsize': 10}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(train_rmse, label='train_rmse', color="#333333")
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/train_rmse.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()

    # alpha
    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot(1, 1, 1)
    plt.title(str("Test Alpha"), fontweight="bold", color="#333333",
              pad=10, fontname='Times New Roman', fontdict={'fontsize': 10})
    # x y labels
    # plt.xlabel('Epochs',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # plt.ylabel('Alpha',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.plot(np.sum(alpha1, 0), label="alpha", color="#333333")
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/alpha.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()

    # alpha bar
    plt.title(str("Test Alpha"), fontweight="bold", color="#333333",
              pad=10, fontname='Times New Roman', fontdict={'fontsize': 10})
    # x y labels
    # plt.xlabel('Epochs',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # plt.ylabel('Alpha',fontweight="bold", color="#333333", fontname = 'Times New Roman', fontdict={'fontsize':18}, labelpad=8)
    # x y ticks
    plt.yticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.xticks(fontweight="bold", color="#333333",
               fontname='Times New Roman', fontsize=8)
    plt.imshow(np.mat(np.sum(alpha1, 0)))
    plt.legend(loc='best', fontsize=8)
    plt.savefig(save_path+'/alpha11.jpg', dpi=500)
    # save the figure
    # plt.show()
    # close the figure
    plt.close()