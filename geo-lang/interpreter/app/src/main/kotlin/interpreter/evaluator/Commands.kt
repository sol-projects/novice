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

fun Polyline(points: List<List<Number>>, name: String, evaluatorInfo: EvaluatorInfo) {
    evaluatorInfo.features.add(createPolylineFeature(points, name))
    evaluatorInfo.featureCollection.add("features", evaluatorInfo.features)
}

fun createPolylineFeature(points: List<List<Number>>, name: String): JsonObject {
    val feature = JsonObject()
    feature.addProperty("type", "Feature")

    val geometry = JsonObject()
    geometry.addProperty("type", "LineString")

    val coordinates = JsonArray()
    for (point in points) {
        coordinates.add(createCoordinate(point[0].toDouble(), point[1].toDouble()))
    }

    geometry.add("coordinates", coordinates)
    feature.add("geometry", geometry)

    val properties = JsonObject()
    properties.addProperty("name", name.substring(1, name.length - 1))
    feature.add("properties", properties)

    return feature
}

fun NPolygon(points: List<List<Number>>, name: String, evaluatorInfo: EvaluatorInfo) {
    evaluatorInfo.features.add(createNPolygonFeature(points, name))
    evaluatorInfo.featureCollection.add("features", evaluatorInfo.features)
}

fun createNPolygonFeature(points: List<List<Number>>, name: String): JsonObject {
    val feature = JsonObject()
    feature.addProperty("type", "Feature")

    val geometry = JsonObject()
    geometry.addProperty("type", "Polygon")

    val coordinates = JsonArray()

    val polygon = JsonArray()
    for (point in points) {
        polygon.add(createCoordinate(point[0].toDouble(), point[1].toDouble()))
    }
    polygon.add(createCoordinate(points[0][0].toDouble(), points[0][1].toDouble()))
    coordinates.add(polygon)

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
    evaluatorInfo.features.add(createCurveFeature(startPointDouble, endPointDouble, degree, name))
    evaluatorInfo.featureCollection.add("features", evaluatorInfo.features)
}

fun createCurveFeature(startPoint: List<Double>, endPoint: List<Double>, degree: Double, name: String): JsonObject {
    val midPointX = (startPoint[0] + endPoint[0]) / 2
    val midPointY = (startPoint[1] + endPoint[1]) / 2

    val deltaX = endPoint[0] - startPoint[0]
    val deltaY = endPoint[1] - startPoint[1]
    val distance = sqrt(deltaX*deltaX + deltaY*deltaY)
    val height = abs(distance * sin(degree * PI / 180.0))

    val curvePoints = mutableListOf<List<Double>>()

    val numberOfPoints = 100
    for (i in 0..numberOfPoints) {
        val t = i.toDouble() / numberOfPoints
        val curveHeight = height * sin(t * PI)
        val point = listOf(midPointX + t * deltaX, midPointY + curveHeight)
        curvePoints.add(point)
    }

    val featureObject = JsonObject()
    featureObject.addProperty("type", "Feature")

    val propertiesObject = JsonObject()
    propertiesObject.addProperty("name", name)
    featureObject.add("properties", propertiesObject)

    val geometryObject = JsonObject()
    geometryObject.addProperty("type", "LineString")

    val coordinatesArray = JsonArray()
    curvePoints.forEach { point ->
        val coordinateArray = JsonArray()
        point.forEach { coordinate ->
            coordinateArray.add(coordinate)
        }
        coordinatesArray.add(coordinateArray)
    }
    geometryObject.add("coordinates", coordinatesArray)

    featureObject.add("geometry", geometryObject)
    return featureObject
}


fun createCoordinate(lon: Double, lat: Double): JsonArray {
    val coordinate = JsonArray()
    coordinate.add(lon)
    coordinate.add(lat)
    return coordinate
}
