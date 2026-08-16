// ======================================================
// 1. KOORDINAT TITIK AWS KEBUMEN
// ======================================================
var point = ee.Geometry.Point([
  109.64610305568304,
  -7.736626058459022
]);

Map.centerObject(point, 10);
Map.addLayer(point, { color: 'red' }, 'Lokasi AWS');

// ======================================================
// 2. DATASET ERA5 HOURLY (Hanya Band yang Dibutuhkan)
// ======================================================
var bandsToSelect = [
  'temperature_2m',
  'dewpoint_temperature_2m',
  'u_component_of_wind_10m',
  'v_component_of_wind_10m',
  'mean_sea_level_pressure',
  'total_precipitation',
  'total_cloud_cover',
  'convective_available_potential_energy',
  'total_column_water_vapour',
  'vertically_integrated_moisture_divergence',
  'total_sky_direct_solar_radiation_at_surface'
];

var dataset = ee.ImageCollection('ECMWF/ERA5/HOURLY')
  .filterDate('1998-01-01', '2026-06-30')
  .select(bandsToSelect);

// ======================================================
// 3. PROSES TRANSFORMASI & PENYESUAIAN FITUR ML
// ======================================================
var era5TS = dataset.map(function (img) {

  var t2m = img.select('temperature_2m');
  var d2m = img.select('dewpoint_temperature_2m');
  var u10 = img.select('u_component_of_wind_10m');
  var v10 = img.select('v_component_of_wind_10m');
  var msl = img.select('mean_sea_level_pressure');
  var tp = img.select('total_precipitation');
  var tcc = img.select('total_cloud_cover');

  // 1. Konversi Temperatur & Dewpoint ke Celsius
  var tempC = t2m.subtract(273.15).rename('temperature');
  var dewC = d2m.subtract(273.15).rename('dewpoint');

  // 2. Perhitungan Relative Humidity (%)
  var es = tempC.multiply(17.625).divide(tempC.add(243.04)).exp();
  var e = dewC.multiply(17.625).divide(dewC.add(243.04)).exp();
  var rh = e.divide(es).multiply(100).clamp(0, 100).rename('humidity');

  // 3. Konversi Tekanan (hPa) & Akumulasi Hujan (mm)
  var pressure = msl.divide(100).rename('pressure');
  var rainrate = tp.multiply(1000).rename('rainrate');

  // 4. Rename Angin & Tutupan Awan
  var uWind = u10.rename('era5_u_wind');
  var vWind = v10.rename('era5_v_wind');
  var cloudCover = tcc.rename('era5_cloud_cover');

  // 5. Estimasi Lamanya Penyinaran Matahari (Proxy)
  var sunshineDuration = ee.Image.constant(3600)
    .multiply(ee.Image.constant(1).subtract(tcc))
    .rename('era5_sunshine');

  // 6. Rename Variabel Atmosferik Lainnya
  var cape = img.select('convective_available_potential_energy').rename('era5_cape');
  var tcwv = img.select('total_column_water_vapour').rename('era5_tcwv');
  var moistureDiv = img.select('vertically_integrated_moisture_divergence').rename('era5_moisture_div');
  var directRad = img.select('total_sky_direct_solar_radiation_at_surface').rename('era5_direct_rad');

  // Menggabungkan Band
  var finalImage = img.addBands([
    tempC, dewC, rh, pressure, rainrate, uWind, vWind, cloudCover,
    sunshineDuration, cape, tcwv, moistureDiv, directRad
  ]);

  // Ekstraksi Nilai Berdasarkan Titik Koordinat AWS
  var values = finalImage.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: 27830,
    maxPixels: 1e13
  });

  var unixTime = img.get('system:time_start');

  return ee.Feature(null, {
    unixtime: unixTime,
    datetime_utc: ee.Date(unixTime).format('YYYY-MM-dd HH:mm:ss'),
    temperature: values.get('temperature'),
    humidity: values.get('humidity'),
    dewpoint: values.get('dewpoint'),
    rainrate: values.get('rainrate'),
    pressure: values.get('pressure'),
    era5_u_wind: values.get('era5_u_wind'),
    era5_v_wind: values.get('era5_v_wind'),
    era5_cloud_cover: values.get('era5_cloud_cover'),
    era5_cape: values.get('era5_cape'),
    era5_tcwv: values.get('era5_tcwv'),
    era5_moisture_div: values.get('era5_moisture_div'),
    era5_direct_rad: values.get('era5_direct_rad'),
    era5_sunshine: values.get('era5_sunshine')
  });
});

// ======================================================
// 4. EKSPOR DATA TABULAR
// ======================================================
Export.table.toDrive({
  collection: era5TS,
  description: 'ERA5_Hourly_All_Requested_Features_2000_2026',
  fileFormat: 'CSV',
  selectors: [
    'unixtime',
    'datetime_utc',
    'temperature',
    'humidity',
    'dewpoint',
    'rainrate',
    'pressure',
    'era5_u_wind',
    'era5_v_wind',
    'era5_cloud_cover',
    'era5_cape',
    'era5_tcwv',
    'era5_moisture_div',
    'era5_direct_rad',
    'era5_sunshine'
  ]
});