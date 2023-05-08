import * as Db from '../db/db';
import dotenv from 'dotenv';
dotenv.config();

test('geocoding', async () => {
  const coords = await Db.Util.toCoords('Maribor');
  expect(
    coords[1] > 45 && coords[1] < 47 && coords[0] > 15 && coords[0] < 16
  ).toBeTruthy();
});

test('reverse geocoding', async () => {
  const coords = await Db.Util.toCoords('Maribor');
  const city = await Db.Util.fromCoords(coords);
  expect(city).toBe('Maribor');
});

test('settlement exists', () => {
  expect(Db.Util.isSettlement('Maribor')).toBeTruthy;
  expect(Db.Util.isSettlement('false_name')).toBeFalsy;
});

test('first settlement in array', () => {
  expect(
    Db.Util.getFirstSettlement(['dsdds', 'Maribor', 'Ljubljana']) == 'Maribor'
  ).toBeTruthy;
  expect(Db.Util.getFirstSettlement(['dsdds', 'asdaass', 'widwajsk']) == '')
    .toBeTruthy;
});
