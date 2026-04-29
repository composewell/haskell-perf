-- |
-- Module      : Streamly.KeyValue.Type
-- Copyright   : (c) 2021 Composewell Technologies
-- License     : Apache-2.0
-- Maintainer  : streamly@composewell.com
-- Stability   : experimental
-- Portability : GHC
--
module Streamly.KeyValue.Type
    (
    -- * KeyValue
      KeyValue (..)
    , Zip (..)
    )
where

import Control.Exception (assert)
import Prelude hiding (zip, zipWith)

-- XXX Use the approach like in https://hackage.haskell.org/package/keys? Or we
-- could use a lens based approach? However, the more we abstract the less
-- comprehensible it becomes.
--
-- XXX We could also use a columnar storage (arrays) to store the keys and
-- values.  Further, keys could be stored in boxed arrays whereas the values
-- could be stored in unboxed arrays.

-- | A key value represents a sample of some value labeled with a unique key of
-- type k and having a value of type v.
--
data KeyValue k v = KeyValue !k !v

-------------------------------------------------------------------------------
-- Instances
-------------------------------------------------------------------------------

instance Functor (KeyValue k) where
    fmap f (KeyValue k v) = KeyValue k (f v)

-- | Append two key values using the Semigroup instance of the underlying
-- value.
--
instance (Eq k, Semigroup v) => Semigroup (KeyValue k v) where
    {-# INLINE (<>) #-}
    -- XXX Only KeyValue having the same key make sense to be combined.
    -- However, matching the keys at run time would incur some cost depending
    -- on the comparison function.
    KeyValue k1 v1 <> KeyValue k2 v2 =
        assert (k1 == k2) (KeyValue k1 (v1 <> v2))

-- Functors that can be Zipped.
class Functor f => Zip f where
  zipWith :: (a -> b -> c) -> f a -> f b -> f c
  zipWith f a b = uncurry f <$> zip a b

  zip :: f a -> f b -> f (a, b)
  zip = zipWith (,)

  -- | Zip applicative
  zap :: f (a -> b) -> f a -> f b
  zap = zipWith id

instance Eq k => Zip (KeyValue k) where
    zipWith f (KeyValue k1 v1) (KeyValue k2 v2) =
        assert (k1 == k2) $ KeyValue k1 (f v1 v2)
