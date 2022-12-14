# hd_analysis
Analyse a large set of human design features with basic statistics and timerseries.

For a given set of human design calculations the following features are analyzed:
- typ
- inner authority
- profile
- split
- active chakras
- gates
- active channels
- inc_cross
- inc_cross-typ

The graphs are plotted into an interactive dashboard via Bokeh library in html format.

## Features

Basic statistics 
- pie chart of all values (one-hot encoded)

Timeseries
- stacked line charts of grouped values (one-hot encoded and grouped by time unit (e.g. year))
 
### Birth rate distribution

If birth rate distribution is known this can be mapped on the native result distribution for comparison.

An example csv-file is provided (data from US [1],[2])

## Example
A Code Example is available in ipynb format (jupyter-notebook)


![Dashboard](https://github.com/MicFell/hd_analysis/blob/main/Dashboard_example.PNG)

### Abbreviation used in data analysis

 - typ = energy typ
   - G = Generator
   - MG = manifesting Generator
   - P = Projector
   - M = Manifestor
   - R = Reflector
- split = split of channel connections
- auth = inner authority
   - SP = solar plexus authority
   - SL = sacral authority
   - SN = splenn authority
   - HT = ego authority
   - GC = self projected authority
   - HT_GC = ege projected authority
   - outher_auth = none or outer (lunar) authority
- chakras
   - HD = head
   - AA = Anja
   - TT = throat
   - GC = G-center
   - HT = heart
   - SL = sacral
   - SN = spleen
   - SP = solar plexus
- inc_cross = incarnation cross
   - RAC = right angle cross
   - LAC = left angle cross
   - JXC = juxtaposition cross
 
#### References
<a id="1">[1]</a> 
https://github.com/amankharwal/Birthrate-Analysis/blob/master/births.csv

<a id="2">[2]</a> 
https://github.com/fivethirtyeight/data/tree/master/births
