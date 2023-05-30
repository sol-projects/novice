package interpreter.evaluator
import kotlin.math.*
import com.google.gson.Gson
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import java.lang.Math.pow
fun createCircleFeature(centerLon: Double, centerLat: Double, radius: Double, name: String): JsonObject {
    val feature = JsonObject()
    feature.addProperty("type", "Feature")

    val geometry = JsonObject()
    geometry.addProperty("type", "Polygon")

    val coordinates = JsonArray()

    val numPoints = 360
    val angleStep = 360.0 / numPoints

    val points = JsonArray()
    for (i in 0 until numPoints) {
        val angle = Math.toRadians(i * angleStep)
        val pointLon = centerLon + radius * cos(angle)
        val pointLat = centerLat + radius * sin(angle)
        val point = JsonArray()
        point.add(pointLon)
        point.add(pointLat)
        points.add(point)
    }

    val firstPoint = JsonArray()
    val firstPointCoordinates = points[0].asJsonArray
    firstPoint.add(firstPointCoordinates[0])
    firstPoint.add(firstPointCoordinates[1])
    points.add(firstPoint)

    coordinates.add(points)
    geometry.add("coordinates", coordinates)
    feature.add("geometry", geometry)

    val properties = JsonObject()
    properties.addProperty("name", name.substring(1, name.length - 1))
    feature.add("properties", properties)

    return feature
}

fun Circle(center: List<Number>, radius: Double, name: String, evaluatorInfo: EvaluatorInfo) {
    evaluatorInfo.features.add(createCircleFeature(center[0].toDouble(), center[1].toDouble(), radius, name))
    evaluatorInfo.featureCollection.add("features", evaluatorInfo.features)
}


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

fun Line(a: List<Number>, b: List<Number>, name: String, evaluatorInfo: EvaluatorInfo) {
    evaluatorInfo.features.add(createLineFeature(a, b, name))
    evaluatorInfo.featureCollection.add("features", evaluatorInfo.features)
}

fun createLineFeature(a: List<Number>, b: List<Number>, name: String): JsonObject {
    val feature = JsonObject()
    feature.addProperty("type", "Feature")

    val geometry = JsonObject()
    geometry.addProperty("type", "LineString")

    val coordinates = JsonArray()
    coordinates.add(createCoordinate(a[0].toDouble(), a[1].toDouble()))
    coordinates.add(createCoordinate(b[0].toDouble(), b[1].toDouble()))

    geometry.add("coordinates", coordinates)
    feature.add("geometry", geometry)

    val properties = JsonObject()
    properties.addProperty("name", name.substring(1, name.length - 1))
    feature.add("properties", properties)

    return feature
}

fun Curve(startPoint: List<Number>, endPoint: List<Number>, degree: Double, name: String, evaluatorInfo: EvaluatorInfo) {
    val startPointDouble = startPoint.map { it.toDouble() }
    val endPointDouble = endPoint.map { it.toDouble() }
    //evaluatorInfo.features.add(createBezierCurveFeature(startPointDouble, endPointDouble, degree.toInt(), name))
    //evaluatorInfo.featureCollection.add("features", evaluatorInfo.features)
}


fun createCoordinate(lon: Double, lat: Double): JsonArray {
    val coordinate = JsonArray()
    coordinate.add(lon)
    coordinate.add(lat)
    return coordinate
}
