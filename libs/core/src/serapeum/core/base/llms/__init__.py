"""
LLM base package: abstract interfaces and shared models for LLM backends.

- Functionalities inside the `core/base/llms/types`, and `core/base/llms/base` are being exported from the core/llms,
so to import any type,
use the following import statement:

from serapeum.core.llms import <type_name>

For example:
```python
>> from serapeum.core.llms import MessageRole
```

- To import any functionality from the other modules `utils`, you can use the following import statement:

from serapeum.core.base.llms.utils import <functionality_name>

For example:
```python
>> from serapeum.core.base.llms.utils import <functionality_name>
```

"""
