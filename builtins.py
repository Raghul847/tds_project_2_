import _ast
import _sitebuiltins
import _typeshed
import sys
import types
from _collections_abc import dict_items, dict_keys, dict_values
from _typeshed import (
    AnnotationForm,
    ConvertibleToFloat,
    ConvertibleToInt,
    FileDescriptorOrPath,
    OpenBinaryMode,
    OpenBinaryModeReading,
    OpenBinaryModeUpdating,
    OpenBinaryModeWriting,
    OpenTextMode,
    ReadableBuffer,
    SupportsAdd,
    SupportsAiter,
    SupportsAnext,
    SupportsDivMod,
    SupportsFlush,
    SupportsIter,
    SupportsKeysAndGetItem,
    SupportsLenAndGetItem,
    SupportsNext,
    SupportsRAdd,
    SupportsRDivMod,
    SupportsRichComparison,
    SupportsRichComparisonT,
    SupportsWrite,
)
from collections.abc import (
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    MutableSet,
    Reversible,
    Set as AbstractSet,
    Sized,
)
from io import BufferedRandom, BufferedReader, BufferedWriter, FileIO, TextIOWrapper
from os import PathLike
from types import CellType, CodeType, GenericAlias, TracebackType

# Importing certain types from collections.abc in builtins.pyi can crash mypy
from typing import (  # noqa: Y022,UP035
    IO,
    Any,
    BinaryIO,
    ClassVar,
    Generic,
    Mapping,
    MutableMapping,
    MutableSequence,
    Protocol,
    Sequence,
    SupportsAbs,
    SupportsBytes,
    SupportsComplex,
    SupportsFloat,
    SupportsIndex,
    TypeVar,
    final,
    overload,
    TYPE_CHECKING,   # <-- changed from type_check_only
    Optional,
    Type,
    Union,
)

# mypy crashes if Literal is imported from typing: see #11247
from typing_extensions import (
    Concatenate,
    Literal,
    LiteralString,
    ParamSpec,
    Self,
    TypeAlias,
    TypeGuard,
    TypeIs,
    TypeVarTuple,
    deprecated,
)

if sys.version_info >= (3, 14):
    from _typeshed import AnnotateFunc

# Type variables
_T = TypeVar("_T")
_I = TypeVar("_I")
_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)
_R_co = TypeVar("_R_co", covariant=True)
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_S = TypeVar("_S")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")
_T5 = TypeVar("_T5")
_SupportsNextT_co = TypeVar("_SupportsNextT_co", bound=SupportsNext[Any], covariant=True)
_SupportsAnextT_co = TypeVar("_SupportsAnextT_co", bound=SupportsAnext[Any], covariant=True)
_AwaitableT = TypeVar("_AwaitableT", bound=Awaitable[Any])
_AwaitableT_co = TypeVar("_AwaitableT_co", bound=Awaitable[Any], covariant=True)
_P = ParamSpec("_P")

# Slice type variables
_StartT_co = TypeVar("_StartT_co", covariant=True)
_StopT_co = TypeVar("_StopT_co", covariant=True)
_StepT_co = TypeVar("_StepT_co", covariant=True)

# Fix: Use `Union` instead of `|` in type variable defaults (mypy < 1.0 does not accept `|` in TypeVar default)
# Also note: `|` syntax is valid in Python 3.10+, but TypeVar default must be a type, not an expression like `A | B`
# So we use `Union` for safety and compatibility.
# Alternatively, just use `Any` as default if flexibility is acceptable.

# Corrected: Use `Union` in contexts where `|` may not be allowed in older mypy versions
from typing import Union  # Needed for Union usage below

# Class definitions

class object:
    __doc__: Optional[str]
    __dict__: dict[str, Any]
    __module__: str
    __annotations__: dict[str, Any]

    @property
    def __class__(self) -> Type[Any]: ...
    
    @__class__.setter
    def __class__(self, value: Type[Any], /) -> None: ...

    def __init__(self) -> None: ...
    
    def __new__(cls) -> Any: ...

    def __setattr__(self, name: str, value: Any, /) -> None: ...
    def __delattr__(self, name: str, /) -> None: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __format__(self, format_spec: str, /) -> str: ...
    def __getattribute__(self, name: str, /) -> Any: ...
    def __sizeof__(self) -> int: ...

    def __reduce__(self) -> Union[str, tuple[Any, ...]]: ...
    def __reduce_ex__(self, protocol: SupportsIndex, /) -> Union[str, tuple[Any, ...]]: ...

    # Only define __getstate__ if Python >= 3.11
    # if sys.version_info >= (3, 11):
    #     def __getstate__(self) -> object: ...

    def __dir__(self) -> Iterable[str]: ...
    def __init_subclass__(cls) -> None: ...
    
    @classmethod
    def __subclasshook__(cls, subclass: type, /) -> bool: ...


class staticmethod(Generic[_P, _R_co]):
    @property
    def __func__(self) -> Callable[_P, _R_co]: ...
    @property
    def __isabstractmethod__(self) -> bool: ...

    def __init__(self, f: Callable[_P, _R_co], /) -> None: ...

    @overload
    def __get__(self, instance: None, owner: type, /) -> Callable[_P, _R_co]: ...
    @overload
    def __get__(self, instance: _T, owner: Optional[type[_T]] = None, /) -> Callable[_P, _R_co]: ...

    if sys.version_info >= (3, 10):
        __name__: str
        __qualname__: str
        @property
        def __wrapped__(self) -> Callable[_P, _R_co]: ...
        def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ...

    if sys.version_info >= (3, 14):
        def __class_getitem__(cls, item: Any, /) -> GenericAlias: ...
        __annotate__: Optional[AnnotateFunc]  # <-- changed from AnnotateFunc | None


class classmethod(Generic[_T, _P, _R_co]):
    @property
    def __func__(self) -> Callable[Concatenate[type[_T], _P], _R_co]: ...
    @property
    def __isabstractmethod__(self) -> bool: ...

    def __init__(self, f: Callable[Concatenate[type[_T], _P], _R_co], /) -> None: ...

    @overload
    def __get__(self, instance: _T, owner: Optional[type[_T]] = None, /) -> Callable[_P, _R_co]: ...
    @overload
    def __get__(self, instance: None, owner: type[_T], /) -> Callable[_P, _R_co]: ...

    if sys.version_info >= (3, 10):
        __name__: str
        __qualname__: str
        @property
        def __wrapped__(self) -> Callable[Concatenate[type[_T], _P], _R_co]: ...

    if sys.version_info >= (3, 14):
        def __class_getitem__(cls, item: Any, /) -> GenericAlias: ...
        __annotate__: Optional[AnnotateFunc]  # <-- changed from AnnotateFunc | None


class type:
    @property
    def __base__(self) -> Optional[type]: ...
    __bases__: tuple[type, ...]
    @property
    def __basicsize__(self) -> int: ...
    @property
    def __dict__(self) -> types.MappingProxyType:  # type: ignore[override]
    @property
    def __dictoffset__(self) -> int: ...
    @property
    def __flags__(self) -> int: ...
    @property
    def __itemsize__(self) -> int: ...
    __module__: str
    @property
    def __mro__(self) -> tuple[type, ...]: ...
    __name__: str
    __qualname__: str
    @property
    def __text_signature__(self) -> Optional[str]: ...
    @property
    def __weakrefoffset__(self) -> int: ...

    @overload
    def __init__(self, name: str, bases: tuple[type, ...], dict: dict[str, Any], /, **kwds: Any) -> None: ...
    @overload
    def __new__(cls, o: object, /) -> type: ...
    @overload
    def __new__(
        cls: Type[Any], name: str, bases: tuple[type, ...], namespace: dict[str, Any], /
    ) -> Any: ...
