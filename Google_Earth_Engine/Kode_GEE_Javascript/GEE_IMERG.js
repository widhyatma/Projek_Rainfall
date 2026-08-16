// ======================================================
// 1. TITIK KOORDINAT AWS KEBUMEN
// ======================================================
var point = ee.Geometry.Point([
    109.64610305568304,
    -7.736626058459022
]);

// ======================================================
// 2. DATASET IMERG (JALUR AMAN: Band yang Pasti Ada di Semua Tahun)
// ======================================================
var bandsToSelect = [
    'precipitation', // Selalu ada di era mana pun (Sinyal Utama)
    'randomError'    // Selalu ada di era mana pun (Margin Eror)
];

var dataset = ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
    .select(bandsToSelect)
    .filterDate('1998-01-01', '2026-07-31');

// ======================================================
// 3. EKSTRAKSI TIME SERIES
// ======================================================
var rainfall = dataset.map(function (img) {

    var value = img.reduceRegion({
        reducer: ee.Reducer.first(),
        geometry: point,
        scale: 10000,
        maxPixels: 1e13
    });

    var unixTime = img.get('system:time_start');
    var datetimeUTC = ee.Date(unixTime).format('YYYY-MM-dd HH:mm:ss');

    return ee.Feature(null, {
        unixtime: unixTime,
        datetime_utc: datetimeUTC,

        // Hanya mengekstrak 2 properti inti yang dijamin stabil
        precipitation: value.get('precipitation'),
        randomError: value.get('randomError')
    });
});

// ======================================================
// 4. PREVIEW DATA DI CONSOLE
// ======================================================
print('Jumlah data ditemukan:', rainfall.size());
print('Struktur data jalur aman:', rainfall.limit(10));

// ======================================================
// 5. EXPORT CSV (Nama File/Description Dikunci)
// ======================================================
Export.table.toDrive({
    collection: rainfall,
    description: 'Rainfall_IMERG_TimeSeries_UNIX', // Nama file tetap dikunci
    fileFormat: 'CSV',
    selectors: [
        'unixtime',
        'datetime_utc',
        'precipitation',
        'randomError'
    ]
});