# NASA GIBS satellite imagery (report figure only)

The report's timeline (a draggable divider between the model's map and the satellite image) and its card cover show a true-colour satellite image. Source: NASA
Global Imagery Browse Services (GIBS), layer `VIIRS_SNPP_CorrectedReflectance_TrueColor` (Suomi NPP VIIRS, daily
corrected-reflectance mosaic), fetched through the public WMS endpoint `https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi`
(no key). Terms: NASA data and imagery are freely available; the report credits "NASA EOSDIS GIBS". See
https://www.earthdata.nasa.gov/engage/open-data-services-software/earthdata-developer-portal/gibs-api and
https://www.earthdata.nasa.gov/earth-observation-data/data-use-policy.

The images are resampled onto the model's 375 m grid by `report/scripts/satellite.py` and are display-only: no model
input, training or metric uses them. Provenance (layer, date rule, area) is in `report/bundle/site.json` under
`timeline.satellite`.
