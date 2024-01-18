package com.mygdx.game.lang

import com.badlogic.gdx.graphics.Mesh
import com.badlogic.gdx.graphics.VertexAttribute
import com.badlogic.gdx.graphics.VertexAttributes
import com.badlogic.gdx.graphics.glutils.ShapeRenderer
import com.mygdx.game.Location
import com.mygdx.game.utils.Geolocation
import com.mygdx.game.utils.MapRasterTiles
import com.mygdx.game.utils.ZoomXY
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import java.io.File
import javax.print.attribute.standard.Media
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.wololo.geojson.FeatureCollection

fun load(): String {
    return File("in.txt").readText()
}

fun code_prepare_to_json(code: String) : String {
    val escape_quotations = code
        .replace("\"", "\\\"")
        //.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t");
    return "{\n\"code\":\n\"${escape_quotations}\"\n}"
}

fun get_geojson_from_interpreter(json: String): String {
    try {
        val client = OkHttpClient()
        val mediaType: MediaType? = "application/json".toMediaTypeOrNull()
        val body = RequestBody.create(mediaType, code_prepare_to_json(json))
        val request: Request = Request.Builder()
            .url("http://localhost:8000/news/geolang")
            .post(body)
            .addHeader("Content-Type", "application/json")
            .build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            println(response.body!!.string())
            throw RuntimeException("Error executing geolang request")
        }
        val responseData = response.body!!.string()
        return responseData
    } catch (e: Exception) {
        e.printStackTrace()
    }

    return ""
}

fun get_points_from_geojson(json: String): ArrayList<Geolocation> {
    val output = arrayListOf<Geolocation>()

    try {
        val jsonObj = Json.parseToJsonElement(json).jsonObject

        if (jsonObj.containsKey("features")) {
            val featuresArray = jsonObj["features"]?.jsonArray
            println("Getting features.")

            featuresArray?.forEach { feature ->
                val geometry = feature.jsonObject["geometry"]?.jsonObject

                // Check if the geometry is of type "Point" or "Polygon"
                val geometryType = geometry?.get("type")?.jsonPrimitive?.contentOrNull
                when (geometryType) {
                    "Point" -> {
                        val coordinates = geometry["coordinates"]?.jsonArray
                        val latitude = coordinates?.get(1)?.jsonPrimitive?.doubleOrNull
                        val longitude = coordinates?.get(0)?.jsonPrimitive?.doubleOrNull

                        if (latitude != null && longitude != null) {
                            output.add(Geolocation(latitude, longitude))
                        }
                    }

                    "Polygon" -> {
                        val coordinates = geometry["coordinates"]?.jsonArray

                        coordinates?.forEach { ring ->
                            ring?.jsonArray?.forEach { point ->
                                val latitude = point?.jsonArray?.get(1)?.jsonPrimitive?.doubleOrNull
                                val longitude =
                                    point?.jsonArray?.get(0)?.jsonPrimitive?.doubleOrNull

                                if (latitude != null && longitude != null) {
                                    output.add(Geolocation(latitude, longitude))
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch (e: Exception) {
        e.printStackTrace()
    }

    return output
}

fun create_point_mesh(shapeRenderer: ShapeRenderer, code_points: ArrayList<Geolocation>, beginTile: ZoomXY): Mesh {
    val pointMesh = Mesh(true, code_points.size, 0,
        VertexAttribute(VertexAttributes.Usage.Position, 2, "a_position")
    )

    val vertices = FloatArray(code_points.size * 2)

    for (i in 0 until code_points.size) {
        val marker = MapRasterTiles.getPixelPosition(
            code_points[i].lat,
            code_points[i].lng,
            beginTile.x, beginTile.y)

        vertices[i * 2] = marker.x
        vertices[i * 2 + 1] = marker.y
    }

    pointMesh.setVertices(vertices)
    return pointMesh
}
