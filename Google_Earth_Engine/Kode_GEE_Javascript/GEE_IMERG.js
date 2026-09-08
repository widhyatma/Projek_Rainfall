var point = ee.Geometry.Point([109.64610305568304, -7.736626058459022]);

var dataset = ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
  .select(['precipitation', 'randomError'])
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
    precipitation: value.get('precipitation'),
    randomError: value.get('randomError')
  });
});

Export.table.toDrive({
  collection: rainfall,
  description: 'Rainfall_IMERG_TimeSeries_UNIX',
  fileFormat: 'CSV',
  selectors: ['unixtime', 'datetime_utc', 'precipitation', 'randomError']
});
