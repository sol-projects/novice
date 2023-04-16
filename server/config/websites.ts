

//https://siol.net/pregled-dneva/2023-4-8/
//
export const websites = new Map<string, () => Promise<string>>([
    ['gov novice', require('./scrapers/gov')],
    ['24 ur', require('./scrapers/24ur')],
    ['siol', require('./scrapers/siol')],
    ['delo', require('./scrapers/delo')],
])
