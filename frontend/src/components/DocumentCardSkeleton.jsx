function DocumentCardSkeleton() {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-2xl">

      <div className="p-6 flex justify-between items-center">

        <div className="flex flex-col gap-2">
          <div className="h-6 w-2xl skeleton rounded animate-pulse"></div>
          <div className="h-4 w-28 skeleton rounded animate-pulse"></div>
        </div>

        <div className="flex items-center gap-3">
          <div className="w-10 h-9 skeleton rounded-xl animate-pulse"></div>
          <div className="w-10 h-9 skeleton rounded-xl animate-pulse"></div>
          <div className="w-10 h-9 skeleton rounded-xl animate-pulse"></div>
          <div className="w-4 h-4 skeleton rounded animate-pulse"></div>
        </div>

      </div>

    </div>
  );
}

export default DocumentCardSkeleton;