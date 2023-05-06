import { INews } from '../model/News';

export const websites = new Map<string, (n: number) => Promise<INews[]>>([
  //['gov novice', require('./websites/gov')],
  //['gov novice vlade', require('./websites/gov_vlada')],
  //['24 ur', require('./websites/24ur')],
  //['siol', require('./websites/siol')],
  //['delo', require('./websites/delo')],
  //['mbinfo', require('./websites/mariborinfo')], //done 
  //['ekipa24', require('./websites/ekipa24')], ///iz nekega razloga sam en news screpa
  //['dnevnik', require('./websites/dnevnik')], // mostly done
  //['svet24', require('./websites/svet24')], // spet samo eno
  ['n1info', require('./websites/n1info')],
]);
