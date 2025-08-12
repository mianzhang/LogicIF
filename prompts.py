"""
Prompt templates for LogicIF framework LLM interactions.
"""

prompt_add_stat = """You are an expert Python programmer. I want to enhance a function by adding meaningful, NON-REDUNDANT execution statistics that capture the most crucial aspects of its computational behavior.

**CRITICAL REQUIREMENTS:**
1. **Adaptive number of statistics (1-3)** - Choose the appropriate number based on function complexity
2. **No redundant statistics** - Each must measure something completely different
3. **Meaningful measurements** - Track operations that reveal algorithmic behavior
4. **Perfect integration** - Statistics should be seamlessly woven into the function logic
5. **Return format** - Function must return a tuple: (original_result, stats_dict)

**STATISTICS SELECTION CRITERIA:**
- **Simple functions (1 statistic)**: Basic loops, single operations
- **Moderate functions (2 statistics)**: Multiple loops, conditional logic, data processing
- **Complex functions (3 statistics)**: Advanced algorithms, multiple phases, intricate logic

**EXAMPLES OF GOOD STATISTICS:**
- `iterations`: Loop iterations, recursive calls
- `comparisons`: Element comparisons, condition checks
- `swaps`: Data exchanges, position changes
- `operations`: Arithmetic operations, data manipulations
- `memory_accesses`: Array/list accesses, data retrievals
- `updates`: Variable modifications, state changes

**AVOID REDUNDANT COMBINATIONS:**
iterations + operations (often correlated)
comparisons + swaps (usually related)
memory_accesses + iterations (typically linked)

**FUNCTION TO ENHANCE:**
```python
{function}
```

**INTEGRATION REQUIREMENTS:**
1. **Rename the function to 'f'** - This is mandatory
2. **Initialize statistics** at the beginning with descriptive variable names
3. **Track throughout execution** - Increment counters at appropriate points
4. **Return tuple** - (original_result, {{"stat1_name": value1, "stat2_name": value2, ...}})
5. **Preserve all original logic** - Don't change the core algorithm
6. **Use clear variable names** - Make statistics purpose obvious

**OUTPUT FORMAT:**
```json
{{
    "function": "def f(...):\n    # Initialize statistics\n    stat1_name = 0\n    stat2_name = 0\n    \n    # Original logic with integrated tracking\n    ...\n    stat1_name += 1  # Where appropriate\n    ...\n    \n    return original_result, {{'stat1_name': stat1_name, 'stat2_name': stat2_name}}",
    "stats_keys": ["stat1_name", "stat2_name"]
}}
```

Choose the optimal number and types of statistics that best capture this function's computational essence while avoiding redundancy."""

prompt_get_desc = """You are helping someone understand how to manually process data step-by-step. I need you to create conversational instructions that explain how to work through the algorithm like you're talking someone through it over the phone - detailed enough that they could follow along exactly and get the same results as the code.

**CRITICAL REQUIREMENTS:**
The instructions must be so complete and precise that someone could follow them step-by-step WITHOUT seeing any code and produce the exact same output and statistics as the function would. Think of this like you're giving someone verbal instructions to manually execute the algorithm.

**CONVERSATIONAL STYLE:**
- Use natural language like you're talking to someone: "Now you need to...", "Next, go through...", "At this point..."
- Sound like verbal instructions, not a formal manual
- Be extremely precise about every detail, but use conversational language
- Use connecting words and phrases that make it flow naturally

**STRUCTURE YOUR RESPONSE:**
Break down into three conversational sections:

**INPUTS SECTION:**
- Describe what data they're starting with in natural language
- Explain the format and structure like you're describing it to someone
- Mention any constraints or special properties conversationally

**LOGICS SECTION:**
- Walk through EVERY step of the algorithm like you're guiding someone
- Explain ALL loops, conditions, and operations in natural language
- Cover every variable update, counter increment, and calculation
- Describe the flow and decision points conversationally
- Make sure every significant operation is explained clearly
- Include details about initialization, iteration, and any edge cases

**OUTPUTS SECTION:**
- Describe exactly what result they should end up with
- Explain the format of the final answer conversationally
- Mention any statistics or tracking values they should have

**COMPLETENESS CHECK:**
Make sure your instructions cover:
- Every loop and what happens in each iteration
- Every conditional check and what to do in each case
- Every variable that gets updated and when
- Every calculation and operation step-by-step
- All edge cases and special conditions
- The exact format of the final result

Function to describe:
```python
{function}
```

Return your response in JSON format:
{{
    "inputs": "Conversational description of inputs like you're explaining to someone...",
    "logics": "Step-by-step conversational walkthrough of the entire algorithm, covering every operation, loop, condition, and calculation in natural language...",
    "outputs": "Conversational description of the expected results and format..."
}}"""

prompt_verify_description = """You are helping someone check if a set of conversational instructions are complete enough for manual data processing. I need you to verify whether these step-by-step instructions would allow someone to manually work through the data and get the same results as the code - like you're checking if verbal instructions over the phone would be complete enough for someone to follow along exactly.

**CRITICAL REQUIREMENT**: The instructions must be so complete and precise that someone could follow them step-by-step WITHOUT seeing any code and produce the exact same output and statistics as the function would.

**VERIFICATION APPROACH:**
Think about this like you're helping someone understand whether these conversational instructions are good enough for manual data processing. The instructions should be so clear and complete that someone could:
1. **Follow Every Step**: All the data processing steps are explained like you're talking someone through it
2. **Handle All Cases**: Every condition, loop, and decision point is covered in natural language
3. **Track Everything**: All variable updates, counters, and calculations are explained conversationally
4. **Get Same Results**: Following the instructions would produce identical output and statistics
5. **No Guessing**: Every significant operation is covered so no one has to guess what to do
6. **Natural Flow**: The instructions flow naturally like someone talking through the process

**CONVERSATIONAL STYLE EXPECTATIONS:**
The instructions should use natural language like:
- "Now you need to...", "Next, go through...", "At this point..."
- "What you're going to do is...", "The way this works is..."
- "Start by creating...", "Add 1 to your counter...", "Figure out the middle point..."
- Sound like verbal instructions, not a formal manual
- Be extremely precise about every detail, but use conversational language
- Use connecting words and phrases that make it flow naturally

**FUNCTION CODE:**
```python
{function_code}
```

**CONVERSATIONAL INSTRUCTIONS TO CHECK:**
{description}

**HOW TO VERIFY:**
- Go through the code step by step like you're walking through it with someone
- Check if the conversational instructions cover each operation naturally
- See if someone following these verbal instructions would do exactly what the code does
- Look for any missing steps, unclear parts, or operations not explained conversationally
- Consider if all loops, conditions, calculations, and counter updates are covered in natural language
- Check if the instructions sound like someone talking through the process step-by-step

**COMPLETENESS JUDGMENT:**
- **COMPLETE**: The conversational instructions cover everything needed - someone could follow them and get identical results
- **INCOMPLETE**: Some operations, conditions, or steps are missing or unclear - following the instructions wouldn't match the code's behavior

**RESPONSE STYLE**: Use the same conversational tone in your assessment - explain things like you're talking to someone about what's working well and what might be missing from these verbal instructions.

Return your response in JSON format:
{{
    "desc_is_complete": true/false,
    "reasoning": "Talk through your assessment conversationally, like you're explaining to someone what's working well in these instructions and what might be missing. Use natural language like 'What I notice is...', 'The instructions do a good job of...', 'What's missing is...', 'Someone following these would probably get confused when...'",
    "missing_aspects": ["List specific operations or steps that aren't covered conversationally - describe them in natural language like 'explaining how to update the counter', 'walking through the loop condition', 'describing what to do when the list is empty'"],
    "coverage_percentage": "estimated percentage (0-100) of code operations covered by the conversational instructions"
}}"""

prompt_extract_output = """You are a data extraction expert. Parse the algorithm execution response and extract structured output and statistics.

**TASK**: Extract the main result and statistics from the manual execution response.

**Context:**
Function Description: {algo_description}
Expected Statistics Keys: {stats_keys}
Expected Output Format: {output_format}

**Response to Parse:**
{llm_response}

**EXTRACTION RULES:**
1. **Output Field**: Extract ONLY the main result value
   - NOT the tuple (result, stats)
   - NOT the statistics dictionary
   - Just the primary return value following the expected format: {output_format}
   - Ensure the output matches the expected format exactly
   - Convert data types as needed (e.g., ensure lists are lists, integers are integers)

2. **Stats Field**: Extract statistics as a dictionary
   - Use the expected statistics keys: {stats_keys}
   - Values must be integers
   - If missing, infer from reasoning or set to 0

**EXAMPLES:**
If response says "Final result is [1, 2, 3] with 5 comparisons and 3 swaps":
- Output: [1, 2, 3]
- Stats: {{"comparisons": 5, "swaps": 3}}

If response says "The answer is 42. I counted 10 operations":
- Output: 42
- Stats: {{"operations": 10}}

Return JSON format:
{{
    "output": extracted_main_result,
    "stats": {{"stat_key": integer_value}}
}}""" 


prompt_update_description = """You are helping someone understand how to manually process data step-by-step. I need you to improve the existing instructions based on feedback that identified missing or unclear parts, while maintaining the exact same format and conversational style.

**CRITICAL REQUIREMENT**: Your improved instructions must be so complete and precise that someone could follow them step-by-step WITHOUT seeing any code and produce the exact same output and statistics as the function would.

**FUNCTION CODE:**
```python
{function_code}
```

**CURRENT INSTRUCTIONS:**
{current_description}

**FEEDBACK ON WHAT'S MISSING:**
- Missing or Unclear Parts: {missing_aspects}
- Detailed Feedback: {verification_reasoning}

**IMPORTANT CONSTRAINTS:**
- Don't explain what this is used for in the real world - just focus on the data processing steps
- Don't mention specific problem names or applications
- Treat this as pure data manipulation work
- Use the generic variable names from the code (L, arr, n, m, etc.)

**Natural Language Guidelines (MAINTAIN THE SAME STYLE):**
- Write like you're speaking to someone conversationally
- Use natural transitions like "Now you need to...", "Next, go through...", "At this point..."
- Include phrases like "What you're going to do is...", "The way this works is..."
- Make it sound like verbal instructions, not a formal manual
- Still be extremely precise about every detail, but use conversational language
- Use connecting words and phrases that make it flow naturally
- Include every conditional check, loop, and decision point in natural speech
- Be specific about indexing, bounds, and conditions, but explain them conversationally

**Conversational Style Examples:**
- Instead of "Initialize variable A as an empty list" → "Start by creating an empty list called A"
- Instead of "For each element s in lst" → "Now you're going to go through each item in the list, and for each one..."
- Instead of "Increment cnt_prt by 1" → "Add 1 to your counter called cnt_prt"
- Instead of "Compute mid as lo plus..." → "Figure out the middle point by taking lo and adding..."

**IMPROVEMENT REQUIREMENTS:**
1. **Fix All Missing Parts**: Address every item mentioned in the feedback
2. **Keep Good Parts**: Don't change parts that are already clear and accurate
3. **Complete Coverage**: Make sure every operation, condition, loop, and logic step is explained conversationally
4. **Natural Flow**: Maintain the step-by-step conversational flow like someone talking through the process
5. **Precise Details**: Include all variable operations, counters, and calculations but explain them naturally
6. **All Cases**: Cover all conditional branches and special situations in natural language
7. **Same Format**: Keep the exact three-section structure (inputs, logics, outputs)

**What to cover in your explanation:**
- How to set up your workspace with the right variables
- Step-by-step processing flow in conversational language  
- When to make decisions and what conditions to check
- How to handle different cases and edge situations
- When and how to update your counters and statistics
- How data gets modified as you work through it
- When to stop and what the final steps are

Return your response in the following JSON format with exactly three sections (same as original):
{{
    "inputs": "Describe the data types and structure of what you're working with (like 'a list of numbers' or 'two text strings'). Don't mention what these represent in real-world terms.",
    "logics": "Give detailed, conversational instructions for processing the data step-by-step. Use natural language like you're talking someone through it, with phrases like 'Now you need to...', 'Next, go through...', 'What you do is...'. Be extremely precise about every step, condition, and operation, but explain it in a flowing, conversational way. Include all loops, decisions, calculations, and counter updates. Use the generic variable names from the code. Address all the missing aspects identified in the feedback.",
    "outputs": "Explain what you'll end up with and what each number in your statistics dictionary represents from the work you did."
}}"""


prompt_solve = """You are a function execution expert. I will provide a detailed algorithmic description and input values. Your task is to manually execute the function step-by-step and determine both the output and execution statistics.

**CRITICAL INSTRUCTIONS:**
- Do NOT write any code - work through the logic manually
- Follow the description exactly as written
- Track all statistics mentioned in the description
- Count every operation precisely
- Show your step-by-step reasoning

**Function Description:**
{algo_description}

**Input Values in Order (List):**
{input}

**REQUIRED RESPONSE FORMAT:**
**Reasoning:** [Step-by-step explanation of how you executed the algorithm]
**Output:** [The exact final result]
**Statistics:** [Dictionary with precise counts for each statistic]""" 


prompt_evolve_function = """You are an expert Python programmer. I want you to evolve an existing function to make it MORE LOGICALLY COMPLICATED while maintaining the exact same input signature and core functionality. The evolved function should be significantly more sophisticated in its logic and computational approach.

**CRITICAL REQUIREMENTS:**
1. **Same Input Signature**: The function must accept exactly the same parameters as the original
2. **Same Core Output**: The main result should be equivalent to the original function's output
3. **More Complex Logic**: Add sophisticated algorithmic patterns, advanced data structures, or multi-phase processing
4. **Enhanced Statistics**: Statistics can change to reflect the new complexity (1-3 meaningful stats)
5. **Preserve Function Name**: Keep the function name as 'f'
6. **Return Format**: Must return tuple (result, stats_dict)
7. **ABSOLUTELY NO COMMENTS**: Do NOT write any comments, docstrings, or explanations in the evolved function code. The function must be completely comment-free.

**EVOLUTION STRATEGIES (choose the most appropriate):**
- **Multi-phase processing**: Break the problem into sophisticated stages
- **Advanced data structures**: Use heaps, trees, graphs, or complex mappings
- **Optimized algorithms**: Replace naive approaches with efficient algorithms
- **Dynamic programming**: Add memoization or tabulation for overlapping subproblems
- **Divide and conquer**: Split problem into smaller, more complex subproblems
- **State machines**: Add complex state tracking and transitions
- **Mathematical optimization**: Add advanced mathematical techniques
- **Sophisticated filtering/sorting**: Use multiple criteria or advanced comparison logic

**COMPLEXITY ENHANCEMENT EXAMPLES:**
- Simple loop → Nested loops with complex conditions
- Linear search → Binary search or hash-based lookup
- Basic sorting → Multi-key sorting with custom comparators  
- Direct calculation → Iterative refinement or approximation
- Single pass → Multiple passes with different objectives
- Static logic → Adaptive logic that changes based on input characteristics

**STATISTICS GUIDELINES:**
- Choose 1-3 statistics that reflect the NEW complexity
- Track operations that highlight the sophisticated logic
- Examples: phases_completed, recursive_calls, cache_hits, comparisons, transformations, iterations

**ORIGINAL FUNCTION:**
```python
{original_function}
```

**REQUIREMENTS:**
1. Analyze the original function's core purpose and constraints
2. Design a more sophisticated approach that achieves the same goal
3. Implement complex logic patterns while preserving correctness
4. Add meaningful statistics that capture the new complexity
5. Ensure the evolved function is significantly more algorithmically interesting
6. Test edge cases and maintain robustness
7. **ABSOLUTELY NO COMMENTS**: The evolved function code must be completely comment-free

**OUTPUT FORMAT:**
```json
{{
    "evolved_function": "def f(...):\n    stat1 = 0\n    stat2 = 0\n    \n    complex_logic_here\n    \n    return result, {{'stat1': stat1, 'stat2': stat2}}",
    "stats_keys": ["stat1", "stat2"],
    "evolution_description": "Brief explanation of how the function was made more complex (e.g., 'Added multi-phase processing with dynamic programming', 'Implemented graph-based approach with state tracking')"
}}
```

**IMPORTANT REMINDERS:**
- The evolved function code must contain ZERO comments or explanations
- No # comments, no docstrings, no inline explanations
- Only pure executable code with meaningful variable names
- Focus on algorithmic sophistication, not code documentation

Make the evolved function significantly more sophisticated while maintaining the same essential behavior and input/output contract.""" 

