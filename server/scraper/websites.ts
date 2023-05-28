import { INews } from '../model/News';

const websites = new Map<string, (n: number) => Promise<INews[]>>([
  ['siol', require('./websites/siol')],
  ['gov', require('./websites/gov')],
  ['gov-vlada', require('./websites/gov_vlada')],
  ['24ur', require('./websites/24ur')],
  ['delo', require('./websites/delo')],
  ['mbinfo', require('./websites/mariborinfo')],
  ['rtvslo', require('./websites/rtvSlo')],
  ['sta', require('./websites/servisSta')],
  ['ekipa24', require('./websites/ekipa24')],
  ['dnevnik', require('./websites/dnevnik')], 
  ['svet24', require('./websites/svet24')], 
  //['n1info', require('./websites/n1info')], 
]);

export = websites;
