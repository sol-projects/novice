package interpreter.evaluator
import com.google.gson.Gson
import com.google.gson.JsonArray
import com.google.gson.JsonObject

fun Box(a: List<Number>, b: List<Number>, evaluatorInfo: EvaluatorInfo, name: String) {
    evaluatorInfo.features.add(createPolygonFeature(a, b, name))
    evaluatorInfo.featureCollection.add("features", evaluatorInfo.features)
}

fun Point(a: List<Number>, name: String, evaluatorInfo: EvaluatorInfo) {
    evaluatorInfo.features.add(createPointFeature(a[0].toDouble(), a[1].toDouble(), name))
    evaluatorInfo.featureCollection.add("features", evaluatorInfo.features)
}

fun createPointFeature(lon: Double, lat: Double, name: String): JsonObject {
    val feature = JsonObject()
    feature.addProperty("type", "Feature")

    val geometry = JsonObject()
    geometry.addProperty("type", "Point")

    val coordinates = JsonArray()
    coordinates.add(lon)
    coordinates.add(lat)

    geometry.add("coordinates", coordinates)
    feature.add("geometry", geometry)

    val properties = JsonObject()
    properties.addProperty("name", name.substring(1, name.length-1))
    feature.add("properties", properties)

    return feature
}

fun createPolygonFeature(a: List<Number>, b: List<Number>, name: String): JsonObject {
    val feature = JsonObject()
    feature.addProperty("type", "Feature")

    val geometry = JsonObject()
    geometry.addProperty("type", "Polygon")

    val coordinates = JsonArray()
    val ring = JsonArray()

    ring.add(createCoordinate(a[0].toDouble(), a[1].toDouble()))
    ring.add(createCoordinate(b[0].toDouble(), a[1].toDouble()))
    ring.add(createCoordinate(b[0].toDouble(), b[1].toDouble()))
    ring.add(createCoordinate(a[0].toDouble(), b[1].toDouble()))
    ring.add(createCoordinate(a[0].toDouble(), a[1].toDouble()))

    coordinates.add(ring)

    geometry.add("coordinates", coordinates)
    feature.add("geometry", geometry)

    val properties = JsonObject()
    properties.addProperty("name", name.substring(1, name.length-1))
    feature.add("properties", properties)

    return feature
}

fun createCoordinate(lon: Double, lat: Double): JsonArray {
    val coordinate = JsonArray()
    coordinate.add(lon)
    coordinate.add(lat)
    return coordinate
}