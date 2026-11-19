package com.shadowzero.remote

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

val JSON = Json { ignoreUnknownKeys = true }
val MEDIA_TYPE = "application/json".toMediaType()
val OK = OkHttpClient.Builder()
    .connectTimeout(10, TimeUnit.SECONDS)
    .readTimeout(0, TimeUnit.MILLISECONDS)
    .build()

class Api(var baseUrl: String, var token: String) {

    private fun build(method: String, path: String, body: String? = null): Request {
        val url = baseUrl.trimEnd('/') + path
        val rb = Request.Builder().url(url)
        if (token.isNotEmpty()) rb.header("Authorization", "Bearer $token")
        when (method) {
            "GET" -> rb.get()
            "DELETE" -> rb.delete(if (body != null) body.toRequestBody(MEDIA_TYPE) else null)
            else -> rb.post((body ?: "{}").toRequestBody(MEDIA_TYPE))
        }
        return rb.build()
    }

    suspend fun json(method: String, path: String, body: String? = null): JsonElement =
        withContext(Dispatchers.IO) {
            OK.newCall(build(method, path, body)).execute().use { resp ->
                val text = resp.body!!.string()
                if (!resp.isSuccessful) throw ApiException(resp.code, text)
                if (text.isBlank()) JsonNull else JSON.parseToJsonElement(text)
            }
        }

    suspend fun pair(url: String, pin: String): String {
        baseUrl = url.trimEnd('/')
        val obj = withContext(Dispatchers.IO) {
            OK.newCall(build("POST", "/pair", JSON.encodeToString(JsonObject.serializer(), buildJsonObject { put("pin", pin) })))
                .execute().use { resp ->
                    val text = resp.body!!.string()
                    if (!resp.isSuccessful) throw ApiException(resp.code, text)
                    JSON.parseToJsonElement(text).jsonObject
                }
        }
        token = obj["token"]?.jsonPrimitive?.content ?: throw ApiException(500, "no token in response")
        return token
    }

    suspend fun tools(): JsonElement = json("GET", "/v1/tools")

    suspend fun memoryList(): JsonElement = json("GET", "/v1/memory")

    suspend fun memoryAdd(kind: String, text: String) =
        json("POST", "/v1/memory", JSON.encodeToString(JsonObject.serializer(), buildJsonObject {
            put("kind", kind); put("text", text)
        }))

    suspend fun memoryDelete(id: Long) = json("DELETE", "/v1/memory?id=$id")

    suspend fun enginePark(): JsonElement = json("POST", "/v1/engine/park")
    suspend fun engineResume(): JsonElement = json("POST", "/v1/engine/resume")

    suspend fun sessionsList(): JsonElement = json("GET", "/v1/sessions")
    suspend fun sessionGet(id: String): JsonElement = json("GET", "/v1/sessions/$id")
    suspend fun sessionSave(body: JsonObject): JsonElement =
        json("POST", "/v1/sessions", JSON.encodeToString(JsonObject.serializer(), body))
    suspend fun status(): JsonElement = json("GET", "/v1/status")
    suspend fun models(): JsonElement = json("GET", "/v1/models")

    suspend fun executeTool(name: String, arguments: String): JsonObject =
        json("POST", "/v1/tools/execute", JSON.encodeToString(JsonObject.serializer(), buildJsonObject {
            put("name", name)
            put("arguments", JsonPrimitive(arguments))
        })).jsonObject

    /** OpenAI-style agentic single step: streams SSE, returns content + tool calls + finish reason. */
    suspend fun chatStep(messages: JsonArray, tools: JsonElement, onText: (String) -> Unit): ChatStep =
        withContext(Dispatchers.IO) {
            val payload = buildJsonObject {
                put("messages", messages)
                put("stream", true)
                if (tools.jsonArray.isNotEmpty()) put("tools", tools)
            }
            val req = build("POST", "/v1/chat/completions", JSON.encodeToString(JsonObject.serializer(), payload))
            OK.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) throw ApiException(resp.code, resp.body!!.string())
                val content = StringBuilder()
                val calls = mutableMapOf<Int, ArrayBuilder>()
                var finish: String? = null
                val reader = resp.body!!.source()
                while (true) {
                    val line = reader.readUtf8Line() ?: break
                    if (!line.startsWith("data:")) continue
                    val data = line.substring(5).trim()
                    if (data == "[DONE]") break
                    val ev = JSON.parseToJsonElement(data).jsonObject
                    val choice = ev["choices"]?.jsonArray?.firstOrNull()?.jsonObject ?: continue
                    val delta = choice["delta"]?.jsonObject ?: continue
                    delta["content"]?.jsonPrimitive?.contentOrNull?.let {
                        content.append(it)
                        onText(it)
                    }
                    delta["tool_calls"]?.jsonArray?.forEach { tc ->
                        val t = tc.jsonObject
                        val idx = t["index"]?.jsonPrimitive?.int ?: 0
                        val b = calls.getOrPut(idx) { ArrayBuilder() }
                        t["id"]?.jsonPrimitive?.contentOrNull?.let { b.id = it }
                        t["function"]?.jsonObject?.let { f ->
                            f["name"]?.jsonPrimitive?.contentOrNull?.let { b.name += it }
                            f["arguments"]?.jsonPrimitive?.contentOrNull?.let { b.args += it }
                        }
                    }
                    choice["finish_reason"]?.jsonPrimitive?.contentOrNull?.let { finish = it }
                }
                ChatStep(content.toString(), calls.values.map { ToolCallOut(it.id, it.name, it.args) }, finish)
            }
        }

    class ArrayBuilder { var id: String = "call_" + System.nanoTime(); var name: String = ""; var args: String = "" }
}

class ApiException(val code: Int, override val message: String) : Exception(message)

data class ToolCallOut(val id: String, val name: String, val arguments: String)
data class ChatStep(val content: String, val toolCalls: List<ToolCallOut>, val finishReason: String?)
