import { INews } from './model/News';

export const websites = new Map<string, (n: number) => Promise<INews[]>>([
  ['gov novice', require('./scrapers/gov')],
  ['gov novice vlade', require('./scrapers/gov_vlada')],
  ['24 ur', require('./scrapers/24ur')],
  //['siol', require('./scrapers/siol')],
  //['delo', require('./scrapers/delo')],
]);
