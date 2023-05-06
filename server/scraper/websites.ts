import { INews } from '../model/News';

const websites = new Map<string, (n: number) => Promise<INews[]>>([
  ['siol', require('./websites/siol')],
  ['gov', require('./websites/gov')],
  ['gov-vlada', require('./websites/gov_vlada')],
  ['24ur', require('./websites/24ur')],
  ['delo', require('./websites/delo')],
  ['mbinfo', require('./websites/mariborinfo')], //done
  ['ekipa24', require('./websites/ekipa24')], ///iz nekega razloga sam en news screpa
  ['dnevnik', require('./websites/dnevnik')], // mostly done
  ['svet24', require('./websites/svet24')], // spet samo eno
  ['n1info', require('./websites/n1info')],
]);

export = websites;
