var point = ee.Geometry.Point([109.64610305568304, -7.736626058459022]);
Map.centerObject(point, 8);
Map.addLayer(point, {color: 'red'}, 'Lokasi');

var dataset = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational')
  .select(['hourlyPrecipRate', 'hourlyPrecipRateGC'])
  .filterDate('2005-01-01', '2026-01-01');

var rainfall = dataset.map(function(img) {
  var value = img.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: 10000,
    maxPixels: 1e13
  });
  var unixTime = img.get('system:time_start');
  return ee.Feature(null, {
    unixtime: unixTime,
    datetime_utc: ee.Date(unixTime).format('YYYY-MM-dd HH:mm:ss'),
    hourlyPrecipRate: value.get('hourlyPrecipRate'),
    hourlyPrecipRateGC: value.get('hourlyPrecipRateGC')
  });
});

Export.table.toDrive({
  collection: rainfall,
  description: 'Rainfall_GSMaP_TimeSeries_UNIX',
  fileFormat: 'CSV',
  selectors: ['unixtime', 'datetime_utc', 'hourlyPrecipRate', 'hourlyPrecipRateGC']
});
