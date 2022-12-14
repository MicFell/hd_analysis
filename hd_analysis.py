import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import hd_features as hd
import hd_constants
import itertools
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer

from bokeh.plotting import figure
from bokeh.plotting import show
from bokeh.models import Legend
from bokeh.models import Panel
from bokeh.models import Tabs
from bokeh.models.widgets import DataTable
from bokeh.models.widgets import TableColumn
from bokeh.palettes import GnBu
from bokeh.palettes import GnBu5
from bokeh.palettes import Category20c
from bokeh.palettes import inferno 
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.models import BoxZoomTool
from bokeh.models import WheelZoomTool
from bokeh.models import LassoSelectTool
from bokeh.models import BoxSelectTool
from bokeh.models import ResetTool
from bokeh.models import PanTool
from bokeh.models import TapTool
from bokeh.models import SaveTool
from bokeh.models import HoverTool
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.transform import cumsum
from math import pi

def get_map_fct(file_name,distribution=False):
    """
    take birth distribution and convert to MapFunction
    time is automatically set to midday (12:00:00),
    timezone_offset=0
    Args:
        inputfile: .csv format(date,births)
    Return
        Map_fct(pd.Series)
        timestamp_list(list): timestamp format(year,month,
                                                day,hour,minute,second,tz_offset)
    """
    df = pd.read_csv(file_name)
    timestamp_list = "(" + df["date"]+",12,0,0,0)"
    timestamp_list = list(
        timestamp_list.apply(lambda x: literal_eval(str(x)))
    )
    if distribution==True:
        Map_fct = df["births"].reset_index(drop=True)
    else:
        Map_fct = pd.Series([1]*len(timestamp_list))
    return Map_fct,timestamp_list

def calc_mult_hd_features_birth_dist(timestamp_list,num_cpu=2):
    """take timestamp list from birth distribution and calc hd features 
    Args:
        timestamp_list(list):  timestamp format(year,month,
                                                day,hour,minute,second,tz_offset)
        num_cpu(int): for multiprocessing
    Returns:
        result_lists(dict): keys: "typ","auth","inc_cross","profile"
                                 "split,"date_to_gate_dict","active_chakra"
                                 "active_channel" are packed in list of dicts
    """
    p = Pool(num_cpu)
    result = process_map(
        hd.calc_single_hd_features,timestamp_list,chunksize=num_cpu)
    p.close()
    p.join()
    result_lists = hd.unpack_mult_features(result,full=True) 
    return result_lists #lists are structured as dict

def ohe_hd_timeseries(cat_features,cat_feature_name,timestamp_list,Map_fct,ohe,time_intervall,norm,order):
    """
    one hot encode of categorial features
    time units supported year,month,day,hour or continous
    Args:
        cat_features(list,or nested lists): hd features from result_lists
        cat_feature_name(str): category name
        timestamp_list(list(tuples): list of given timestamps for encoding
        map_fct(pd.Series): known birth distribution can be applied here
        unit(str):data may grouped by time unit, or not grouped -> continous
        ohe(str): different encoding methods, for each feature. 
                  Input format is variable (e.g. nested lists,sets..)      
    Return:
        df_ohe(dataframe): encoded dataframe (normalization is not performed,but can be..) 
    """
    #calc of time unit list (grouped or continous)
    if (time_intervall == "continous") |  (time_intervall [:6] == "custom"):
        time_unit_list = timestamp_list
    else:
        time_unit_dict = {"years":0,"months":1,"days":2,"hours":3}
        time_unit = time_unit_dict[time_intervall] 
        time_unit_list = [timestamp_list[i][time_unit] 
                          for i in range(len(timestamp_list))]
    #one hot encode methode, depenting on input format
    if ((ohe == "profile") | (ohe == "split") | (ohe == "auth") | (ohe == "typ") 
        | (ohe == "inc_cross") | (ohe == "inc_cross_typ")) :
        if (ohe == "split"):
            cat_features = [str(elem) for elem in cat_features] 
        df_ohe = pd.DataFrame({cat_feature_name:cat_features}, index = time_unit_list)
        df_ohe = pd.get_dummies(df_ohe) 
        
    elif ohe == "active_channel":   
        #zip gates and ch_gate to channel
        s =[list(zip((cat_features[i]["gate"]),(cat_features[i]["ch_gate"]))) for i in range(len(cat_features))]
        #fast one-hot-encoding for list of lists
        mlb = MultiLabelBinarizer()
        df_ohe = pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=time_unit_list)
        #remove duplicates (1,2)=(2,1)
        sorted_cols = [tuple(sorted(col)) for col in df_ohe.columns]
        sorted_cols,df_ohe.columns
        df_ohe.columns = sorted_cols
        df_ohe = df_ohe.groupby(axis=1, level=0).sum()
                
    elif (ohe == "gate") | (ohe == "active_chakra"):        
        #fast one-hot-encoding for list of lists
        mlb = MultiLabelBinarizer()
        s=pd.Series(cat_features)
        df_ohe = pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=time_unit_list)  
        if (order == "rave") & (ohe == "gate"):
            df_ohe = df_ohe[hd_constants.IGING_CIRCLE_LIST]            
    #apply map function-> elementwise multiplication (need to drop index, and to recreate the old one)
    df_ohe = df_ohe.reset_index(drop=True).mul(Map_fct,axis=0).set_index(df_ohe.index)
   
    if time_intervall == "continous":
        df_ohe_result = df_ohe
    elif time_intervall [:6] == "custom":
        datetime_format = [','.join(map(str, elem[:-1])) 
                        for elem in df_ohe.index]
        date_index = pd.to_datetime(datetime_format,format='%Y,%m,%d,%H,%M,%S')
        df_ohe.index = date_index
        df_ohe_result = df_ohe.groupby(pd.Grouper(freq=time_intervall[7:])).sum()
    else:
        df_ohe_result = df_ohe.groupby(axis=0, level=0).sum()

    if norm:
        df_ohe_result = df_ohe_result.div(df_ohe_result.sum(axis=1), axis=0).fillna(0) #normalisation 

    return df_ohe_result

def ohe_hd_features(result_lists,hd_feature_list,timestamp_list,Map_fct,time_intervall,norm=False,order="sorted"):
    """one hotencode multiple data to df and pack to dict
    Args:
        result_lists(dict): input data (stacked in dict)
        hd_feature_list(list): list of features that should be encoded
        timestamp_list(list(tuples): list of given timestamps for encoding
        time_intervall(str): continous or grouped by unit(years,months,days,hours are supported)
        Map_fct(pd.Series): known birth distribution can be applied here
    Return:
        data_dict(dict):encoded df's stored in dict
                        keys: "typ","auth","inc_cross","profile"
                              "split,"date_to_gate_dict","active_chakra"
                              "active_channel"
    """
    ohe_data_dict = {}
    fig_dict = {}
    
    for elem in tqdm(hd_feature_list):
        raw_data = result_lists["{}_list".format(elem)]
        df = ohe_hd_timeseries(raw_data,elem,timestamp_list,Map_fct,elem,time_intervall,norm,order)
        ohe_data_dict[elem] = df
        
    return ohe_data_dict

def get_hd_stat_graph(result_lists,ohe_data_dict,hd_feature_list,fig_size,Map_fct):
    """
       perform basic statistic-> pie chart on average values (normalized)
       or ranked lists if to many categories
       plot charts interactiv with bokeh
       Args:
           result_lists(dict): input data (stacked in dict)
           data_dict(dict):one hot encoded df's packed in dict
           hd_feature_list(list): list of features that should be encoded
           fig_size(list): e.g. [100,100], size of each widget
           Map_fct(pd.Series): known birth distribution can be applied here
       Return:
           bokeh grid
    """       
    fig_dict = {}
    for elem in hd_feature_list:
        #for single category data 
        if elem in ["typ","auth","profile","split","inc_cross_typ"]: 
            raw_data = pd.Series(result_lists["{}_list".format(elem)])
            raw_data.name = elem
            data=pd.concat([raw_data, Map_fct], axis=1).groupby(elem).sum()
            data = round(data/data.sum()*100,2) #normalization
            data.columns = ["value"]
        else: #and one hot encoded data (multi cat)
            raw_data = ohe_data_dict[elem]
            data = raw_data.reset_index(drop=True).mul(Map_fct,axis=0).sum().reset_index()
            data.columns = [elem,"value"]
            data["value"] = round(data["value"]/data.value.sum()*100,2)
        #more than 20 categories -> list, less than 20 -> pie chart
        if (elem == "active_channel") | (elem == "gate") | (elem == "inc_cross"):
            raw_data = ohe_data_dict[elem]
            data = raw_data.reset_index(drop=True).mul(Map_fct,axis=0).sum().reset_index()
            data.columns = [elem,"value"]
            data["value"] = round(data["value"]/data.value.sum()*100,2)     
            columns_A = [TableColumn(field=elem, title=elem), TableColumn(field="value", title="count in %") ]       
            data = data.sort_values(by=["value"],ascending=False)
            source_A = ColumnDataSource(data)
            p = DataTable(source=source_A, columns=columns_A,  reorderable=False, width=fig_size[0], height=fig_size[1])
        
        else:
            data['angle'] = data['value']/data['value'].sum() * 2*pi 
            data['color'] = suit_color_palette(len(data))
            p = figure(height=350, title="{}-Distribution".format(elem), toolbar_location=None,
                       tools="hover", tooltips="@{}: @value %".format(elem), x_range=(-0.5, 1))
            p.wedge(x=0, y=1, radius=0.4,
                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                    line_color="white", fill_color='color', legend_field=elem, source=data)
            p.axis.axis_label = None
            p.axis.visible = False
            p.grid.grid_line_color = None

        fig_dict["{}_fig".format(elem)] = p
    grid = gridplot([fig for fig in fig_dict.values()], ncols=2, width=fig_size[0], height=fig_size[1])
    
    return grid


def suit_color_palette(length_data):
    """
    get suitable color palette for given number of categories
    Args:
        length(int): how many colors are needed
    Return:
        color(list):bokeh color palette
    """
    if (length_data < 3):
        colors = (GnBu5[0:length_data])
    elif (length_data <= 5) & (length_data >= 3):
        colors = itertools.cycle(GnBu[length_data])
        colors = tuple([next(colors) for m in range(length_data)])
    elif (length_data > 5) & (length_data <= 20):
        colors = itertools.cycle(Category20c[length_data])
        colors = tuple([next(colors) for m in range(length_data)])
    elif length_data >20:
        colors = itertools.cycle(inferno(length_data))
        colors = tuple([next(colors) for m in range(length_data)])
    return colors

def get_interactive_hd_timeseries(df_ohe_result,time_intervall,Width,Height):
    """
    make timeseries of one hot encoded data with bokeh
    Args:
        df_ohe(pd.DataFrame): used data
        time_intervall(str): continous or grouped by unit(e.g. year,month,day)
        Map_fct(pd.Series): known birth distribution can be applied here
        Width(int): size of widget
        Height(int): size of widget
    Return:
        bokeh figure
    """
    length_data = len(df_ohe_result.columns)
    colors = suit_color_palette(length_data)
   
    raw_data_stacked = {
        str(col):list(df_ohe_result.iloc[:,df_ohe_result.columns.get_loc(col)]) 
        for col in df_ohe_result.columns
    }
    data_stacked = raw_data_stacked.copy()
    
    if time_intervall == "continous":
        datetime_format = [','.join(map(str, elem[:-1])) 
                           for elem in df_ohe_result.index]
        data_stacked["time"] = pd.to_datetime(datetime_format,format='%Y,%m,%d,%H,%M,%S')
    else:
        data_stacked["time"] = df_ohe_result.index 
        
    source = ColumnDataSource(data=data_stacked)
    
    Tools = [PanTool(), BoxZoomTool(match_aspect=True), WheelZoomTool(), BoxSelectTool(),
            ResetTool(), TapTool(), SaveTool()]
    
    if (time_intervall == "continous") | (time_intervall[:6] == "custom"):
        tooltips_list = [('time', '@time{%F}'),("Value", "$y")]
        Tools.append(
            HoverTool(tooltips=tooltips_list,formatters={'@time': 'datetime'}))
        p = figure(
            width=Width, height=Height, toolbar_location="below", tools=Tools,x_axis_type='datetime')
    else:
        tooltips_list = [("time","@time"),("Value", "$y")]
        Tools.append(HoverTool(tooltips=tooltips_list))
        p = figure( width=Width, height=Height, toolbar_location="below", tools=Tools)

    rs = p.varea_stack([str(key) 
                        for key in raw_data_stacked.keys()],color=colors, x='time',source=source)

    Labels=[str(elem) for elem in df_ohe_result.columns]
    legend = Legend(items=[(Labels, [r]) 
                           for (Labels, r) in zip(Labels, rs)], location=(0, 30))
    p.toolbar.logo = None
    p.add_layout(legend, 'right')
    p.legend.click_policy="hide"
    
    return p

def get_tabbed_hd_ts(data_dict,time_intervall,hd_feature_list,ohe_size_dict):
    """
    make bokeh tabs from each figure
    Args:
        df_ohe(dict): used data (encoded data, packed in dict)
        time_intervall(str): continous or grouped by unit(e.g. year,month,day)
        hd_feature_list(list): list of features that should be encoded
        ohe_dict_size(dict): figure size of each figure packed in dict
    Return:
        bokeh tab packed in dict
    """
    tab_dict = {}
    for elem in tqdm(hd_feature_list):
        p = get_interactive_hd_timeseries(data_dict[elem],time_intervall,ohe_size_dict[elem][0],ohe_size_dict[elem][1])
        tab_dict["{}_tab".format(elem)] = Panel(child = p, title = "Timeseries {}".format(elem))

    return tab_dict