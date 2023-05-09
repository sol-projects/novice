import mongoose from 'mongoose';
import settlements from './settlements';

export function connect(name: string) {
  if (!process.env) {
    console.error(
      'You must call dotenv.config() before calling this function.'
    );
  } else {
    mongoose.connect(
      `mongodb+srv://${process.env.DB_USERNAME}:${process.env.DB_PASSWORD}@cluster0.bd0tfwp.mongodb.net/${name}?retryWrites=true&w=majority`
    );
  }
}

export namespace Util {
  export async function toCoords(place: string): Promise<[number, number]> {
    const url = `https://geokeo.com/geocode/v1/search.php?q=${place}&api=${process.env.GEOKEO_API_KEY}`;
    const response = await fetch(url);
    const data = await response.json();
    const results = data.results;

    if (!data.results) {
      return [0, 0];
    }

    const slovenianCities = results.filter((result: any) => {
      return (
        result.address_components.country === 'Slovenia' &&
        result.class === 'place' &&
        result.type === 'city'
      );
    });

    if (slovenianCities.length === 0) {
      return [
        Number(results[0].geometry.location.lng),
        Number(results[0].geometry.location.lat),
      ];
    }

    return [
      Number(slovenianCities[0].geometry.location.lng),
      Number(slovenianCities[0].geometry.location.lat),
    ];
  }

  export async function fromCoords(coords: [number, number]): Promise<string> {
    const url = `https://geokeo.com/geocode/v1/reverse.php?lat=${coords[1]}&lng=${coords[0]}&api=${process.env.GEOKEO_API_KEY}`;
    const response = await fetch(url);
    const data = await response.json();
    const results = data.results;

    if (!data.results) {
      return '';
    }

    for (const result of results) {
      if (result.type == 'city') {
        return result.address_components.name;
      }
    }

    return '';
  }

  export function getFirstSettlement(settlementsArray: string[]) {
    for (const settlement of settlementsArray) {
      if (isSettlement(settlement)) {
        return settlement;
      }
    }

    return '';
  }

  export function isSettlement(settlement: string) {
    return settlement in settlements;
  }
}
