var point = ee.Geometry.Point([
  109.64610305568304,
  -7.736626058459022
]);
Map.centerObject(point, 8);
Map.addLayer(point, { color: 'red' }, 'Lokasi');

var dataset = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational')
  .select([
    'hourlyPrecipRate',
    'hourlyPrecipRateGC'
  ])
  // Rentang waktu penelitian: 1 Januari 2005 s/d 31 Desember 2025
  // Batas akhir diset '2026-01-01' karena filterDate bersifat end-exclusive [start, end)
  .filterDate('2005-01-01', '2026-01-01');

// TIME SERIES EXTRACTION
var rainfall = dataset.map(function (img) {
  var value = img.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: 10000,
    maxPixels: 1e13
  });
  // TIME STANDARD (UTC)
  var unixTime = img.get('system:time_start');
  var datetimeUTC = ee.Date(img.get('system:time_start'))
    .format('YYYY-MM-dd HH:mm:ss');
  return ee.Feature(null, {
    unixtime: unixTime,
    datetime_utc: datetimeUTC,
    hourlyPrecipRate: value.get('hourlyPrecipRate'),
    hourlyPrecipRateGC: value.get('hourlyPrecipRateGC')
  });
});

// PREVIEW DATA
print('Jumlah data:', rainfall.size());
print('Contoh data:', rainfall.limit(10));

// EXPORT CSV
Export.table.toDrive({
  collection: rainfall,
  description: 'Rainfall_GSMaP_TimeSeries_UNIX',
  fileFormat: 'CSV',
  selectors: [
    'unixtime',
    'datetime_utc',
    'hourlyPrecipRate',
    'hourlyPrecipRateGC'
  ]
});