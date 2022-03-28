################################################################################
#
# Provide a dataclass which automatically tries to cast any type-hinted field
# to the type hint. It also as provides an abstract method
# for constructing the object it configures.
#
# Author(s): Nik Vaessen
################################################################################

import dataclasses

from enum import Enum
from typing import TypeVar, Generic, Union

################################################################################
# base configuration which supports casting to type hint and provides abstract
# interface for creating an object based on the configuration

C = TypeVar("C")


@dataclasses.dataclass()
class CastingConfig(Generic[C]):
    def __post_init__(self):
        post_init_type_cast(self)


def post_init_type_cast(dataclass):
    if not dataclasses.is_dataclass(dataclass):
        raise Exception("Can only type-cast dataclass classes.")

    for field in dataclasses.fields(dataclass):
        value = getattr(dataclass, field.name)
        typehint_cls = field.type

        if value is None:
            # no value specified to type-convert
            continue

        # support Optional[t] and Optional[List[t]]
        elif (
            hasattr(typehint_cls, "__origin__")
            and typehint_cls.__origin__ == Union
            and typehint_cls.__args__[1] is type(None)
        ):
            # optional value, but not None (as we passed first check)
            nested_typehint_class = typehint_cls.__args__[0]

            if (
                hasattr(nested_typehint_class, "__origin__")
                and nested_typehint_class.__origin__ == list
            ):
                typehint_cls = nested_typehint_class.__args__[0]
                obj = [typehint_cls(v) for v in value]
            else:
                obj = typehint_cls(value)

        # support List[t]
        elif hasattr(typehint_cls, "__origin__") and typehint_cls.__origin__ == list:
            typehint_cls = typehint_cls.__args__[0]
            obj = [typehint_cls(v) for v in value]

        elif isinstance(value, typehint_cls):
            # no need for type-conversion
            continue

        elif isinstance(value, dict):
            """
            if execution gets here, we know
            value is not an instance of typehinted-type but
            is a dictionary. It contains the contents
            of a nested dataclass
            """
            obj = typehint_cls(**value)

            # recursively perform type casting
            post_init_type_cast(obj)

        elif issubclass(typehint_cls, Enum):
            # enum's have a different init procedure
            obj = typehint_cls[value]

        else:
            # simply type-cast the object
            obj = typehint_cls(value)

        setattr(dataclass, field.name, obj)
