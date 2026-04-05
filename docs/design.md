# langchain moonshot integration

单论对话测试样例

```shell
curl https://api.moonshot.cn/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $MOONSHOT_API_KEY" \
    -d '{
        "model": "kimi-k2.5",
        "messages": [
            {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
            {"role": "user", "content": "你好，我叫李雷，1+1等于多少？"}
        ]
   }'
```

response:

```json
{
    "id": "chatcmpl-69d0ae30d0a91d3e5fce21f1",
    "object": "chat.completion",
    "created": 1775283760,
    "model": "kimi-k2.5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "你好，李雷！很高兴认识你。\n\n在数学中，**1+1 等于 2**。\n\n这是最基础的算术运算结果。当然，如果在其他特定场景下（比如二进制计算、字符串拼接等），结果可能会不同，但通常我们默认指的是十进制加法，答案就是 **2**。",
                "reasoning_content": "用户用中文问候，并介绍自己叫李雷（这似乎是引用中国英语教材中的经典人物），然后问了一个简单的数学问题\"1+1等于多少？\"。\n\n我需要：\n1. 礼貌地回应问候\n2. 确认对方的名字\n3. 回答数学问题\n\n关于1+1：\n- 在标准算术中，1+1=2\n- 但也可以提及其他情况（二进制中1+1=10，或者作为字符串拼接\"1\"+\"1\"=\"11\"），不过对于普通用户问这个问题，通常期待的是标准答案2\n\n考虑到用户用中文提问，我应该用中文回答。\n\n回答结构：\n1. 问候回应：\"你好，李雷！\"\n2. 直接回答：1+1=2\n3. 可以适当友好地补充说明（可选）\n\n避免过度复杂化，保持简洁友好。"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 83,
        "completion_tokens": 243,
        "total_tokens": 326,
        "cached_tokens": 83,
        "prompt_tokens_details": {
            "cached_tokens": 83
        }
    }
}
```

tool_use 样例

```shell
curl https://api.moonshot.cn/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $MOONSHOT_API_KEY" \
    -d '{
        "model": "kimi-k2.5",
        "messages": [
            {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
            {"role": "user", "content": "编程判断 3214567 是否是素数。"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "CodeRunner",
                "description": "代码执行器，支持运行 python 和 javascript 代码",
                "parameters": {
                    "properties": {
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript"]
                        },
                        "code": {
                            "type": "string",
                            "description": "代码写在这里"
                        }
                    },
                "type": "object"
                }
            }
        }]
   }'
```

response

```json
{
    "id": "chatcmpl-69d0af28dc2dfbd2237bd63e",
    "object": "chat.completion",
    "created": 1775284008,
    "model": "kimi-k2.5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "用户想要判断3214567是否是素数。我需要编写代码来检查这个数字是否是素数。\n\n素数是只能被1和它本身整除的大于1的自然数。\n\n检查一个数n是否是素数的方法：\n1. 如果n \u003c= 1，不是素数\n2. 如果n == 2，是素数\n3. 如果n是偶数，不是素数\n4. 检查从3到sqrt(n)的所有奇数，看是否能整除n\n\n让我编写Python代码来检查3214567是否是素数。",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "CodeRunner:0",
                        "type": "function",
                        "function": {
                            "name": "CodeRunner",
                            "arguments": "{\"code\": \"\\nimport math\\n\\ndef is_prime(n):\\n    \\\"\\\"\\\"判断一个数是否是素数\\\"\\\"\\\"\\n    if n \u003c= 1:\\n        return False\\n    if n == 2:\\n        return True\\n    if n % 2 == 0:\\n        return False\\n    # 只需检查到 sqrt(n)\\n    for i in range(3, int(math.sqrt(n)) + 1, 2):\\n        if n % i == 0:\\n            return False, i  # 返回 False 和找到的因数\\n    return True, None\\n\\nnumber = 3214567\\nresult, factor = is_prime(number)\\n\\nif result:\\n    print(f\\\"{number} 是素数\\\")\\nelse:\\n    print(f\\\"{number} 不是素数\\\")\\n    if factor:\\n        print(f\\\"它可以被 {factor} 整除\\\")\\n        print(f\\\"{number} = {factor} × {number // factor}\\\")\\n\", \"language\": \"python\"}"
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }
    ],
    "usage": {
        "prompt_tokens": 146,
        "completion_tokens": 354,
        "total_tokens": 500,
        "cached_tokens": 146,
        "prompt_tokens_details": {
            "cached_tokens": 146
        }
    }
}
```

```
ChatCompletion(
    id='chatcmpl-69d1e257ac12d3c3a34a1b5f',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content=(
                    "Here are your calculations:\n\n"
                    "**3 + 4 = 7**\n\n"
                    "**5 - 6 = -1**\n\n"
                    "**1233 × 4567 = 5,631,111**\n\n"
                    "And here's a joke for you:\n\n"
                    "*Parallel lines have so much in common. It's a shame they'll never meet.*"
                ),
                refusal=None,
                role='assistant',
                annotations=None,
                audio=None,
                function_call=None,
                tool_calls=None,
                reasoning_content=(
                    "The user wants me to perform several calculations and tell a joke:\n\n"
                    "1. Add 3 + 4\n"
                    "2. Subtract 5 - 6\n"
                    "3. Multiply 1233 * 4567\n"
                    "4. Tell a joke\n\n"
                    ...
                    ...
                    "Structure the response clearly."
                )
            )
        )
    ],
    created=1775362647,
    model='kimi-k2.5',
    object='chat.completion',
    service_tier=None,
    system_fingerprint=None,
    usage=CompletionUsage(
        completion_tokens=616,
        prompt_tokens=35,
        total_tokens=651,
        completion_tokens_details=None,
        prompt_tokens_details=None
    )
)
```

## 需要修改的点

参照deepseek

但是deepseek所需的返回格式是纯粹字符串的，moonshot的适配应该是在原本chatopenai基础上，增加reasoning_content，对此需要增加reasoning_content相关的流式处理内容

测试

```
{'model': 'kimi-k2.5', 'stream': False, 'temperature': 1.0, 'tools': [{'type': 'function', 'function': {'name': 'add', 'description': 'This is an addition function that adds 2 numbers together.', 'parameters': {'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b'], 'type': 'object'}}}, {'type': 'function', 'function': {'name': 'subtract', 'description': 'This is a subtraction function that subtracts 2 numbers.', 'parameters': {'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b'], 'type': 'object'}}}, {'type': 'function', 'function': {'name': 'multiply', 'description': 'This is a multiplication function that multiplies 2 numbers.', 'parameters': {'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b'], 'type': 'object'}}}], 'messages': [{'content': 'You are my AI assistant, please answer my query to the best of your ability.', 'role': 'system'}, {'content': 'Add 3 + 4, subtract 5 - 6, multiply 1233 * 4567, Also tell me a joke', 'role': 'user'}]}
```